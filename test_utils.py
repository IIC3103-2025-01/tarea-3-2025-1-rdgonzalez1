# test_utils.py

import numpy as np
import time
import requests
from src.utils import generate_embedding_nomic
from src.config import EMBED_ENDPOINT, REQUEST_TIMEOUT

def test_embeddings_differ_and_shape():
    # 1. Primer vector
    v1 = generate_embedding_nomic("foo")
    # 2. Segundo vector con diferente texto
    v2 = generate_embedding_nomic("bar")

    # Comprueba que la forma es (768,)
    assert isinstance(v1, list) and len(v1) == 768, f"v1 tiene shape {np.shape(v1)}, se esperaba (768,)"
    assert isinstance(v2, list) and len(v2) == 768, f"v2 tiene shape {np.shape(v2)}, se esperaba (768,)"

    # Comprueba que v1 y v2 no sean idénticos
    norm_diff = np.linalg.norm(np.array(v1) - np.array(v2))
    assert norm_diff > 0, "Los embeddings de 'foo' y 'bar' son idénticos, deberían diferir"

    print("✔ Embeddings shape OK y difieren entre sí (norm diff = %.4f)" % norm_diff)

def test_timeout_returns_zeros(monkeypatch):
    # 1. Forzamos un timeout en requests.post
    def fake_post(*args, **kwargs):
        raise requests.exceptions.Timeout("Simulated timeout")
    monkeypatch.setattr(requests, "post", fake_post)

    # 2. Llamada tras timeout: debería devolver lista de ceros
    v = generate_embedding_nomic("anything")
    assert isinstance(v, list) and len(v) == 768, "Tras timeout, debe retornar lista de longitud 768"
    assert all(x == 0.0 for x in v), "Tras timeout, todos los valores deben ser 0.0"

    print("✔ Timeout manejado correctamente, devuelve ceros")

if __name__ == "__main__":
    print("Testeando generate_embedding_nomic…\n")
    test_embeddings_differ_and_shape()
    test_timeout_returns_zeros()
    print("\nTodos los tests pasaron correctamente.")
