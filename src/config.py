import os

# --- Rutas de carpetas ---
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, "data")
DOCS_DIR       = os.path.join(DATA_DIR, "docs")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")

FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_index.index")
METADATA_PATH    = os.path.join(EMBEDDINGS_DIR, "metadatos.json")

# --- Endpoints y parámetros de la API ---
EMBED_ENDPOINT   = "https://asteroide.ing.uc.cl/api/embed"
EMBED_MODEL      = "nomic-embed-text"

CHAT_ENDPOINT    = "https://asteroide.ing.uc.cl/v1/chat/completions"
CHAT_MODEL       = "integracion"

# — Parámetros FAISS y text-splitting —
VECTOR_DIM    = 768
TOP_K         = 6
CHUNK_SIZE    = 100    
CHUNK_OVERLAP = 20     


REQUEST_TIMEOUT     = 30   # segundos
MAX_TOKENS_CONTEXT  = 512  # Límite de tokens para Llama3.2 (integracion)
