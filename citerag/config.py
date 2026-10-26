"""Central settings. Override any value via environment variables or .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_DIR = Path(os.getenv("CITERAG_DB_DIR", ROOT_DIR / ".chroma"))

# LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("CITERAG_LLM_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = int(os.getenv("CITERAG_MAX_TOKENS", "1024"))

# Embeddings / vector store
# bge-small reads up to 512 tokens (all-MiniLM-L6-v2 silently truncates at 256).
EMBED_MODEL = os.getenv("CITERAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_MAX_TOKENS = int(os.getenv("CITERAG_EMBED_MAX_TOKENS", "512"))
# bge models expect this prefix on queries (not on documents).
EMBED_QUERY_PREFIX = os.getenv(
    "CITERAG_EMBED_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: ",
)
COLLECTION_NAME = os.getenv("CITERAG_COLLECTION", "citerag")

# Chunking — all sizes are measured in the embedding model's own tokens.
CHUNK_SIZE = int(os.getenv("CITERAG_CHUNK_SIZE", "384"))  # target max per chunk
CHUNK_OVERLAP = int(os.getenv("CITERAG_CHUNK_OVERLAP", "64"))  # whole sentences
CHUNK_MIN = int(os.getenv("CITERAG_CHUNK_MIN", "96"))  # merge/avoid smaller

# Retrieval
TOP_K = int(os.getenv("CITERAG_TOP_K", "5"))
MIN_SCORE = float(os.getenv("CITERAG_MIN_SCORE", "0.3"))
