import faiss
import numpy as np
from src.utils import generate_embedding_nomic

def test_faiss_index_search():
    # Genera dos embeddings
    v_foo = np.array(generate_embedding_nomic("foo"), dtype="float32")
    v_bar = np.array(generate_embedding_nomic("bar"), dtype="float32")

    dim = v_foo.shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.stack([v_foo, v_bar], axis=0))

    # Busca usando foo
    D, I = index.search(v_foo.reshape(1, -1), k=2)
    # I[0][0] debería ser 0 (foo mismo), y I[0][1] 1 (bar)
    assert I[0][0] == 0
    assert I[0][1] == 1
    # La primera distancia debe ser ~0
    assert D[0][0] < 1e-6

    print("✔ FAISS index funciona correctamente para 2 vectores básicos")
