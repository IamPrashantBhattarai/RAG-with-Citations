# CiteRAG

Document Q&A with verifiable source citations. Upload PDFs/notes, ask questions, and get answers where every claim links back to the exact file, page, and passage.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Run

```bash
streamlit run app.py
pytest
```

## Layout

| Path | Purpose |
|---|---|
| `citerag/config.py` | Settings (model, chunk size, top-k) |
| `citerag/loaders.py` | Load PDF/TXT/MD into page records |
| `citerag/chunker.py` | Split pages into chunks with stable IDs |
| `citerag/store.py` | Embeddings + ChromaDB |
| `citerag/retriever.py` | Top-k retrieval |
| `citerag/generator.py` | Claude answer with `[n]` citations |
| `citerag/citations.py` | Citation parsing + validation |
| `app.py` | Streamlit UI |
| `eval/` | Evaluation question set + scoring |
| `data/` | Your source documents (git-ignored) |
