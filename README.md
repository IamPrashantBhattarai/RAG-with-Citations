# CiteRAG

Document Q&A with verifiable source citations. Upload PDFs/notes, ask questions, and get answers where every claim links back to the exact file, page, and passage.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add ANTHROPIC_API_KEY, or GEMINI_API_KEY for the free Gemini backend
```

## Run

```bash
streamlit run app.py   # opens http://localhost:8501 (this computer only)
pytest
```

In the app: add PDF/TXT/MD files in the sidebar, then ask a question. Each answer lists its sources;
open one to see the quoted passage and the cited PDF page. Statements their sources don't support
are highlighted. Strict mode removes them instead.

## Evaluate

The question set in `eval/questions.json` covers the three PDFs it lists, which must be in `data/`:

```bash
python -m eval.retrieval                 # retrieval + "not in the documents" gate, local, ~1 min
python -m eval.answers --tag mytag       # full pipeline with the LLM; resumable, --recheck re-runs checks only
```

Results and the choices they led to: `ai_agents/08-evaluation.md`.

## Layout

| Path | Purpose |
|---|---|
| `citerag/config.py` | Settings (model, chunk size, top-k) |
| `citerag/loaders.py` | Load PDF/TXT/MD into page records |
| `citerag/chunker.py` | Split pages into chunks with stable IDs |
| `citerag/store.py` | Embeddings + ChromaDB |
| `citerag/retriever.py` | Hybrid retrieval (dense + BM25 keyword search, fused, then reranked) with a "not in the documents" gate |
| `citerag/generator.py` | Cited answers with Claude (native Citations: exact quotes per claim) |
| `citerag/gemini.py` | Free Gemini backend for answers |
| `citerag/citations.py` | Parse [n] citation markers into exact quotes |
| `citerag/validator.py` | Check each statement against what it cites (uncited claims, numbers/names, local NLI model); strict mode drops unsupported ones |
| `citerag/ui.py` | UI helpers: safe answer rendering, quote highlighting, uploads, cited PDF pages |
| `app.py` | Streamlit UI |
| `eval/` | Phase 8 evaluation: question set with exact-quote gold labels, retrieval and answer benchmarks |
| `data/` | Your source documents (git-ignored) |
