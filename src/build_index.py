import os
import numpy as np
import faiss
from tqdm import tqdm
import json

from src.config import (
    DOCS_DIR,
    EMBEDDINGS_DIR,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    VECTOR_DIM
)
from src.utils import (
    split_text_to_chunks,
    generate_embedding_nomic,
    save_metadata,
    load_metadata
)

def build_or_load_faiss_index(path: str, dim: int) -> faiss.IndexFlatL2:
    if os.path.exists(path):
        index = faiss.read_index(path)
        print(f"[FAISS] Cargado índice existente desde '{path}'.")
    else:
        index = faiss.IndexFlatL2(dim)
        print(f"[FAISS] Creando nuevo índice de dimensión {dim}.")
    return index

def main():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    index = build_or_load_faiss_index(FAISS_INDEX_PATH, VECTOR_DIM)

    metadatos = load_metadata(METADATA_PATH)
    existente_ids = {(m["doc_id"], m["chunk_id"]) for m in metadatos}

    archivos = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".txt")]
    print(f"[Index] Encontrados {len(archivos)} archivos en '{DOCS_DIR}'.")
    nuevos_vectores = []
    nuevos_metadatos = []

    for archivo in archivos:
        doc_id = os.path.splitext(archivo)[0]
        ruta_txt = os.path.join(DOCS_DIR, archivo)
        with open(ruta_txt, "r", encoding="utf-8") as f:
            texto = f.read()

        chunks = split_text_to_chunks(texto)
        print(f"[Index] Procesando '{archivo}' → {len(chunks)} fragmentos.")

        for idx, fragmento in enumerate(tqdm(chunks, desc=f"Chunks {archivo}")):
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
        print(f"[FAISS] Se agregaron {len(nuevos_vectores)} vectores nuevos al índice.")

        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"[FAISS] Índice guardado en '{FAISS_INDEX_PATH}'.")

        metadatos.extend(nuevos_metadatos)
        save_metadata(metadatos, METADATA_PATH)
        print(f"[Index] Metadatos guardados en '{METADATA_PATH}' (total {len(metadatos)}).")
    else:
        print("[Index] No se encontraron fragmentos nuevos para indexar.")

if __name__ == "__main__":
    main()
