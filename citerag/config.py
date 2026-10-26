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
LLM_MODEL = os.getenv("CITERAG_LLM_MODEL", "claude-opus-5-5")
# Includes thinking tokens (always on for Opus 5.5); only what's used is billed.
MAX_TOKENS = int(os.getenv("CITERAG_MAX_TOKENS", "16000"))
# low | medium | high | xhigh | max. Grounded Q&A over a few excerpts rarely needs more.
LLM_EFFORT = os.getenv("CITERAG_LLM_EFFORT", "medium")

# Embeddings / vector store
# bge-base won a 5-model benchmark (see ai_agents/03-vector-store.md). It reads up
# to 512 tokens and shares bge-small's tokenizer (all-MiniLM-L6-v2 truncates at 256).
EMBED_MODEL = os.getenv("CITERAG_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_MAX_TOKENS = int(os.getenv("CITERAG_EMBED_MAX_TOKENS", "512"))
# bge models expect this prefix on queries (not on documents).
EMBED_QUERY_PREFIX = os.getenv(
    "CITERAG_EMBED_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: ",
)
EMBED_DOC_PREFIX = os.getenv("CITERAG_EMBED_DOC_PREFIX", "")
EMBED_DEVICE = os.getenv("CITERAG_EMBED_DEVICE") or None  # None = cuda > mps > cpu
EMBED_BATCH_SIZE = int(os.getenv("CITERAG_EMBED_BATCH_SIZE", "32"))
COLLECTION_NAME = os.getenv("CITERAG_COLLECTION", "citerag")

# Chunking — all sizes are measured in the embedding model's own tokens.
CHUNK_SIZE = int(os.getenv("CITERAG_CHUNK_SIZE", "384"))  # target max per chunk
CHUNK_OVERLAP = int(os.getenv("CITERAG_CHUNK_OVERLAP", "64"))  # whole sentences
CHUNK_MIN = int(os.getenv("CITERAG_CHUNK_MIN", "96"))  # merge/avoid smaller

# Retrieval
TOP_K = int(os.getenv("CITERAG_TOP_K", "5"))  # chunks handed to the LLM
RETRIEVE_CANDIDATES = int(os.getenv("CITERAG_RETRIEVE_CANDIDATES", "30"))  # per search, reranked
# Hybrid search alone beat every reranker tested (ai_agents/04-retrieval.md).
USE_RERANKER = os.getenv("CITERAG_USE_RERANKER", "0") == "1"
RERANK_MODEL = os.getenv("CITERAG_RERANK_MODEL", "Alibaba-NLP/gte-reranker-modernbert-base")
# If the best dense match (cosine) is below this, the question is treated as not
# covered by the documents. Calibrated for bge-base; re-check if the model changes.
MIN_SCORE = float(os.getenv("CITERAG_MIN_SCORE", "0.45"))
