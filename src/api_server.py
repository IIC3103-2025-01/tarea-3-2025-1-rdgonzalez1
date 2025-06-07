import os
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import faiss
import numpy as np
import json

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

app = FastAPI(
    title="RAG Wikipedia Chatbot",
    version="1.0",
    description="Backend for Task 3: scraping, RAG, LLM integration"
)

class UploadArticleRequest(BaseModel):
    url: str

class UploadArticleResponse(BaseModel):
    status: str
    doc_id: str

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/upload-article", response_model=UploadArticleResponse)
def upload_article(req: UploadArticleRequest):
    url = req.url.strip()
    if not url.lower().startswith("https://en.wikipedia.org/wiki/"):
        raise HTTPException(status_code=400, detail="URL must start with 'https://en.wikipedia.org/wiki/'.")
    try:
        texto = scrape_wikipedia_article(url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping article: {e}")

    doc_id = str(uuid.uuid4())
    filename = f"{doc_id}.txt"
    try:
        save_text_to_file(texto, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

    try:
        index = build_or_load_faiss_index(FAISS_INDEX_PATH, VECTOR_DIM)
        metadatos = load_metadata(METADATA_PATH)

        chunks = split_text_to_chunks(texto)
        nuevos_vectores = []
        nuevos_metadatos = []
        existente_ids = {(m["doc_id"], m["chunk_id"]) for m in metadatos}

        for idx, fragmento in enumerate(chunks):
            key = (doc_id, idx)
            if key in existente_ids:
                continue
            vector = generate_embedding_nomic(fragmento)
            vector_np = np.array(vector, dtype="float32")
            nuevos_vectores.append(vector_np)
            nuevos_metadatos.append({
                "doc_id": doc_id,
                "chunk_id": idx,
                "text": fragmento
            })
            existente_ids.add(key)

        if nuevos_vectores:
            all_vecs = np.stack(nuevos_vectores, axis=0)
            index.add(all_vecs)
            faiss.write_index(index, FAISS_INDEX_PATH)

            metadatos.extend(nuevos_metadatos)
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(metadatos, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing article: {e}")

    return UploadArticleResponse(status="Article indexed successfully", doc_id=doc_id)

@app.post("/query", response_model=QueryResponse)
def query_article(req: QueryRequest):
    pregunta = req.question.strip()
    if not pregunta:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not os.path.exists(FAISS_INDEX_PATH):
        raise HTTPException(status_code=500, detail="FAISS index not found. Please upload an article first.")
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading FAISS index: {e}")

    if not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=500, detail="Metadata not found. Please upload an article first.")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadatos = json.load(f)

    vector_q = generate_embedding_nomic(pregunta)
    vector_q_np = np.array(vector_q, dtype="float32").reshape(1, -1)

    distances, indices = index.search(vector_q_np, TOP_K)
    indices = indices[0]

    context_chunks = []
    for idx in indices:
        if 0 <= idx < len(metadatos):
            context_chunks.append(metadatos[idx]["text"])

    # Filter chunks by presence of question keywords
    keywords = [w.lower() for w in pregunta.split()]
    filtered = [c for c in context_chunks if any(kw in c.lower() for kw in keywords)]
    if not filtered:
        return QueryResponse(answer="No relevant fragments found for your question.")

    try:
        respuesta = chat_completion_rag(filtered, pregunta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating LLM response: {e}")

    return QueryResponse(answer=respuesta)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
