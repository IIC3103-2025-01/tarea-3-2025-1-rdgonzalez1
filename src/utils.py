import os
import json
import re
import requests
import numpy as np

from typing import List, Dict
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import (
    DOCS_DIR,
    EMBED_ENDPOINT,
    EMBED_MODEL,
    CHAT_ENDPOINT,
    CHAT_MODEL,
    VECTOR_DIM,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    REQUEST_TIMEOUT
)

# 1. Scraping de Wikipedia
def scrape_wikipedia_article(url: str) -> str:
    """
    Given a Wikipedia article URL in English, scrape and return the text of all <p> tags
    within the #bodyContent div, removing reference markers like [1], [2], etc.
    """
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"[Scraping] Error downloading URL: {e}")

    soup = BeautifulSoup(resp.text, "lxml")
    content_div = soup.find("div", id="bodyContent")
    if not content_div:
        raise RuntimeError("[Scraping] 'bodyContent' container not found.")

    text = ""
    ref_pattern = re.compile(r"\[\d+\]")
    for p in content_div.find_all("p"):
        paragraph = p.get_text()
        paragraph = ref_pattern.sub("", paragraph)
        text += paragraph + "\n\n"
    return text.strip()

# 2. Guardar texto en .txt
def save_text_to_file(text: str, filename: str) -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)
    filepath = os.path.join(DOCS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

# 3. Dividir texto en fragmentos
def split_text_to_chunks(text: str, 
                         chunk_size: int = CHUNK_SIZE, 
                         chunk_overlap: int = CHUNK_OVERLAP
                        ) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

# 4. Generar embedding con nomic-embed-text
def generate_embedding_nomic(text: str) -> List[float]:
    payload = {"model": EMBED_MODEL, "input": text}
    try:
        resp = requests.post(EMBED_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        print(f"[Embedding] Error generating embedding: {e}")
        return [0.0] * VECTOR_DIM

    data = resp.json()
    if "embeddings" in data and isinstance(data["embeddings"], list):
        return data["embeddings"][0]
    else:
        raise RuntimeError(f"[Embedding] Unexpected response format: {data}")

# 5. Guardar/Cargar metadatos
def save_metadata(metadata: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_metadata(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 6. Llamar a LLM con RAG context
def chat_completion_rag(context_chunks: List[str], question: str) -> str:
    """
    Send the retrieved context chunks and user question to the LLM and return the generated answer.
    The model is instructed to answer strictly using only the provided context fragments.
    """
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant. Answer questions strictly based on the provided text fragments. "
        "Do not add any external information or assumptions. "
        "If the answer is not contained in the fragments, respond that you don't know."
    )
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer concisely:"
    )

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 512
    }

    try:
        resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        print(f"[Chat] Error calling LLM: {e}")
        return "Sorry, there was an error querying the language model."

    data = resp.json()
    if "choices" in data and isinstance(data["choices"], list):
        return data["choices"][0]["message"]["content"].strip()
    else:
        return "Unable to extract a valid response from the language model."
