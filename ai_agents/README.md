# CiteRAG — Build Log

A record of every step taken to build this project: **what** was done and **why** it was done that way. Updated after every code change or system/environment change.

## Files

| File | Covers | Status |
|---|---|---|
| [00-setup.md](00-setup.md) | Project structure, Python environment, dependencies, git | ✅ Done |
| [01-loaders.md](01-loaders.md) | Loading PDF/TXT/MD into page records, text cleanup | ✅ Done |
| [02-chunking.md](02-chunking.md) | Splitting pages into chunks for embedding | ✅ Done |
| [03-vector-store.md](03-vector-store.md) | Embedding model benchmark, embeddings, ChromaDB index | ✅ Done |
| [CHANGELOG.md](CHANGELOG.md) | Dated list of every change | Ongoing |

Upcoming: `04-retrieval.md`, `05-generation.md`, `06-citation-checks.md`, `07-ui.md`, `08-evaluation.md`.

## Roadmap at a glance

| Phase | Goal | Status |
|---|---|---|
| 0 | Setup | ✅ |
| 1 | Document loaders | ✅ |
| 2 | Chunking | ✅ |
| 3 | Embeddings + ChromaDB | ✅ |
| 4 | Retrieval | ⬜ |
| 5 | Cited answers with Claude | ⬜ |
| 6 | Citation validation | ⬜ |
| 7 | Streamlit UI | ⬜ |
| 8 | Evaluation + tuning | ⬜ |

## Key decisions so far

1. **Python 3.12 via `uv`**, not system Python 3.9 — 3.9 is end-of-life and too old for current ChromaDB/Streamlit. ([00-setup.md](00-setup.md))
2. **Embedding model `BAAI/bge-base-en-v1.5`**: first `bge-small` replaced `all-MiniLM-L6-v2` (MiniLM truncates at 256 tokens), then `bge-base` won a 5-model benchmark (best average nDCG@10 and Recall@5, same tokenizer as bge-small). ([02-chunking.md](02-chunking.md), [03-vector-store.md](03-vector-store.md))
3. **Structure-aware sentence packing** for chunking, 384 tokens / 64 overlap, measured in the embedding model's own tokens. ([02-chunking.md](02-chunking.md))
4. **Physical PDF page numbers + PDF cleanup** (headers/footers, hyphenation, inferred paragraph breaks) so citations match the viewer and the chunker can see headings. ([01-loaders.md](01-loaders.md))
5. **ChromaDB, cosine, our own vectors only**, with per-file fingerprints for incremental re-indexing and a model lock that refuses to mix vectors from different models. ([03-vector-store.md](03-vector-store.md))
