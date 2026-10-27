"""Central settings. Override any value via environment variables or .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_DIR = Path(os.getenv("CITERAG_DB_DIR", ROOT_DIR / ".chroma"))

# LLM
# "anthropic" (Claude, native citations) or "gemini" (Google, free tier).
LLM_PROVIDER = os.getenv("CITERAG_LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("CITERAG_LLM_MODEL", "claude-opus-5-5")
# Includes thinking tokens (always on for Opus 5.5); only what's used is billed.
MAX_TOKENS = int(os.getenv("CITERAG_MAX_TOKENS", "16000"))
# low | medium | high | xhigh | max. Grounded Q&A over a few excerpts rarely needs more.
LLM_EFFORT = os.getenv("CITERAG_LLM_EFFORT", "medium")
# Gemini (key: GEMINI_API_KEY). Free tier: Google may use prompts to improve its products.
# gemini-3.6-flash, not the newer 3.8: on 2026-10-26 3.8 returned "high demand" (503)
# to every free-tier request we sent, while 3.6 answered in ~3 s (05-generation.md).
GEMINI_MODEL = os.getenv("CITERAG_GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_THINKING = os.getenv("CITERAG_GEMINI_THINKING", "low")  # minimal | low | medium | high
# Tried in order when the main model is overloaded (503), out of free quota (429) or times out.
GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("CITERAG_GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite").split(",")
    if m.strip()
]

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

# Citation validation (Phase 6)
VALIDATE = os.getenv("CITERAG_VALIDATE", "1") == "1"  # check every answer's statements
STRICT = os.getenv("CITERAG_STRICT", "0") == "1"  # remove statements that fail the checks
# Won a 4-model benchmark on the syllabus, real Gemini answers and RAGTruth
# (ai_agents/06-citation-checks.md). The large variant
# (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) is slightly more
# accurate and about 3x slower.
NLI_MODEL = os.getenv("CITERAG_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
# A statement counts as supported when the NLI model's P(entailment) for its cited
# quote or passage reaches this. Calibrated for the model above.
NLI_THRESHOLD = float(os.getenv("CITERAG_NLI_THRESHOLD", "0.5"))
