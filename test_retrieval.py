# test_retrieval.py

import os
import json
import numpy as np
import tempfile
import shutil
from starlette.testclient import TestClient
from src.api_server import app
from src.config import FAISS_INDEX_PATH, METADATA_PATH, TOP_K
from src.build_index import build_or_load_faiss_index
from src.utils import split_text_to_chunks, generate_embedding_nomic

# Create TestClient with FastAPI app
client = TestClient(app)



def setup_module(module):
    # Prepare a temporary directory for a mini-index
    module.tmpdir = tempfile.mkdtemp()
    module.index_path = os.path.join(module.tmpdir, "test.index")
    module.meta_path = os.path.join(module.tmpdir, "test_meta.json")

    # Two simple documents
    docs = ["apple orange banana", "car bus train"]
    metadata = []

    # Dimension from embedding function (should be 768)
    dim = len(generate_embedding_nomic("foo"))

    # Build or load FAISS index at the temporary path
    index = build_or_load_faiss_index(module.index_path, dim)

    # Insert each chunk from the docs
    for doc_id, text in enumerate(docs):
        chunks = split_text_to_chunks(text, chunk_size=50, chunk_overlap=0)
        for i, chunk in enumerate(chunks):
            vec = np.array(generate_embedding_nomic(chunk), dtype="float32")
            index.add(vec.reshape(1, -1))
            metadata.append({"doc_id": str(doc_id), "chunk_id": i, "text": chunk})

    # Save index and metadata
    import faiss
    faiss.write_index(index, module.index_path)
    with open(module.meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    # Override config paths so the API uses our test index
    from src import config
    config.FAISS_INDEX_PATH = module.index_path
    config.METADATA_PATH    = module.meta_path


def teardown_module(module):
    shutil.rmtree(module.tmpdir)


def test_query_returns_top_k():
    # Query about 'apple'
    resp = client.post("/query", json={"question": "What fruit?"})
    assert resp.status_code == 200
    body = resp.json()
    # Response should include 'answer' and 'chunks'
    assert "answer" in body and "chunks" in body
    # Should retrieve at most TOP_K chunks
    assert len(body["chunks"]) <= TOP_K
    # The most relevant chunk should contain 'apple'
    assert "apple" in body["chunks"][0].lower()


def test_query_no_index():
    # Simulate missing index by deleting files
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)
    resp = client.post("/query", json={"question": "anything"})
    assert resp.status_code == 500
    assert "index not found" in resp.json()["detail"].lower()
