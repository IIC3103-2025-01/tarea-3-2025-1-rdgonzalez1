import os
import uuid
import json
import faiss
import numpy as np
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import (
    DOCS_DIR,
    EMBEDDINGS_DIR,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    VECTOR_DIM,
    TOP_K
)
from src.utils import (
    scrape_wikipedia_article,
    save_text_to_file,
    load_metadata,
    generate_embedding_nomic,
    chat_completion_rag,
    split_text_to_chunks
)
from src.build_index import build_or_load_faiss_index

# ---------------------------------------------------
# Inicializa FastAPI
# ---------------------------------------------------
app = FastAPI(
    title="RAG Wikipedia Chatbot",
    version="1.0",
    description="Backend + Frontend estático integrado"
)

# ---------------------------------------------------
# CORS: permitir llamadas desde tu frontend en producción
# ---------------------------------------------------
# En producción, define la URL de tu frontend aquí o como variable de entorno FRONTEND_URL
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Monta el build de React en "/" (tras npm run build)
# ---------------------------------------------------
build_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "build")
if os.path.isdir(build_dir):
    app.mount("/", StaticFiles(directory=build_dir, html=True), name="static")

# ---------------------------------------------------
# Modelos Pydantic para los endpoints
# ---------------------------------------------------
class UploadArticleRequest(BaseModel):
    url: str

class UploadArticleResponse(BaseModel):
    status: str
    doc_id: str

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# ---------------------------------------------------
# Endpoint: subir y procesar artículo
# ---------------------------------------------------
@app.post("/upload-article", response_model=UploadArticleResponse)
def upload_article(req: UploadArticleRequest):
    url = req.url.strip()
    if not url.lower().startswith("https://en.wikipedia.org/wiki/"):
        raise HTTPException(400, "URL must start with 'https://en.wikipedia.org/wiki/'.")
    try:
        texto = scrape_wikipedia_article(url)
    except Exception as e:
        raise HTTPException(500, f"Error scraping article: {e}")

    doc_id = str(uuid.uuid4())
    filename = f"{doc_id}.txt"
    try:
        save_text_to_file(texto, filename)
    except Exception as e:
        raise HTTPException(500, f"Error saving file: {e}")

    try:
        index = build_or_load_faiss_index(FAISS_INDEX_PATH, VECTOR_DIM)
        metadatos = load_metadata(METADATA_PATH)

        chunks = split_text_to_chunks(texto)
        nuevos_vectores, nuevos_metadatos = [], []
        seen = {(m["doc_id"], m["chunk_id"]) for m in metadatos}

        for idx, frag in enumerate(chunks):
            key = (doc_id, idx)
            if key in seen:
                continue
            vec = np.array(generate_embedding_nomic(frag), dtype="float32")
            nuevos_vectores.append(vec)
            nuevos_metadatos.append({
                "doc_id": doc_id,
                "chunk_id": idx,
                "text": frag
            })
            seen.add(key)

        if nuevos_vectores:
            index.add(np.stack(nuevos_vectores, axis=0))
            faiss.write_index(index, FAISS_INDEX_PATH)
            metadatos.extend(nuevos_metadatos)
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(metadatos, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(500, f"Error indexing article: {e}")

    return UploadArticleResponse(status="Article indexed successfully", doc_id=doc_id)


# ---------------------------------------------------
# Endpoint: consulta RAG
# ---------------------------------------------------
@app.post("/query", response_model=QueryResponse)
def query_article(req: QueryRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "Question cannot be empty.")

    if not os.path.exists(FAISS_INDEX_PATH):
        raise HTTPException(500, "FAISS index not found. Please upload an article first.")
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
    except Exception as e:
        raise HTTPException(500, f"Error loading index: {e}")

    if not os.path.exists(METADATA_PATH):
        raise HTTPException(500, "Metadata not found. Please upload an article first.")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadatos = json.load(f)

    vec_q = np.array(generate_embedding_nomic(q), dtype="float32").reshape(1, -1)
    _, ids = index.search(vec_q, TOP_K)
    chunks = [metadatos[i]["text"] for i in ids[0] if i < len(metadatos)]

    # Filtrado simple por palabras clave
    kws = q.lower().split()
    filtered = [c for c in chunks if any(k in c.lower() for k in kws)]
    if not filtered:
        return QueryResponse(answer="No relevant fragments found for your question.")

    try:
        resp = chat_completion_rag(filtered, q)
    except Exception as e:
        raise HTTPException(500, f"Error calling LLM: {e}")

    return QueryResponse(answer=resp)


# ---------------------------------------------------
# Lanzamiento
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=port, reload=False)
