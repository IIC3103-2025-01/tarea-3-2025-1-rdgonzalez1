# build_mini_index.py
import os, json, faiss, numpy as np
from src.utils import split_text_to_chunks, generate_embedding_nomic
from src.build_index import build_or_load_faiss_index
from src.config import VECTOR_DIM

# Directorio temporal
os.makedirs("temp_test", exist_ok=True)
idx_path = "temp_test/test.index"
meta_path = "temp_test/test_meta.json"

# Frases de ejemplo
docs = ["apple orange banana", "car bus train"]
metadata = []

# Crea/obtiene índice
index = build_or_load_faiss_index(idx_path, VECTOR_DIM)

# Inserta cada chunk
for doc_id, text in enumerate(docs):
    chunks = split_text_to_chunks(text, chunk_size=50, chunk_overlap=0)
    for i, chunk in enumerate(chunks):
        vec = np.array(generate_embedding_nomic(chunk), dtype="float32")
        index.add(vec.reshape(1, -1))
        metadata.append({"doc_id": str(doc_id), "chunk_id": i, "text": chunk})

# Guarda índice y metadatos
faiss.write_index(index, idx_path)
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f)

print("✅ Mini-índice creado en temp_test/")
