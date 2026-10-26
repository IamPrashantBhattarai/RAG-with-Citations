# Changelog

Newest first. Each entry links to the phase doc with the full reasoning.

## 2026-10-26

### Added — Phase 3: Embeddings + ChromaDB ([03-vector-store.md](03-vector-store.md))
- `citerag/store.py`: `Embedder` (sentence-transformers, normalized vectors, query-only prefix, auto CUDA/MPS/CPU) and `Store` (persistent ChromaDB, cosine HNSW, citation metadata, exact chunk text).
- Incremental indexing with per-file fingerprints (text + model + chunk settings); changed files fully replaced; `--prune`, `--reset`, `--stats`, `--query` CLI.
- `EmbeddingMismatchError`: an index built with one model can't be opened with another; `Store(reset=True)` to rebuild.
- `tests/test_store.py`: 12 tests with an offline fake embedder.

### Changed — Embedding model
- `EMBED_MODEL`: `BAAI/bge-small-en-v1.5` → `BAAI/bge-base-en-v1.5`, chosen by benchmark on BEIR SciFact + NFCorpus (bge-base: 0.740 / 0.374 nDCG@10 vs. bge-small 0.713 / 0.343). Same tokenizer, so chunk sizes are unchanged.
- New settings: `EMBED_DOC_PREFIX`, `EMBED_DEVICE`, `EMBED_BATCH_SIZE`.

### System
- Downloaded models to the Hugging Face cache for the benchmark: bge-base, arctic-embed-m-v1.5, gte-modernbert-base, Qwen3-Embedding-0.6B (~2.5 GB total; only bge-base is used).

### Added — Phase 1: Document loaders ([01-loaders.md](01-loaders.md))
- `citerag/loaders.py`: `load_file`, `load_pdf`, `load_text`, `load_dir`, `load_paths`, `LoaderError`.
- Text cleanup for all files (line endings, BOM, no-break spaces, ligatures, zero-width/control characters, whitespace). UTF-8 with a cp1252 fallback.
- PDF cleanup: running header/footer and page-number removal, hyphenated line-break rejoining, paragraph-break inference.
- Blank/scanned pages skipped with warnings; corrupt/encrypted PDFs raise `LoaderError` (skipped with a warning in folder loads).
- Preview CLI: `python -m citerag.loaders <files or folders>`.
- `tests/test_loaders.py`: 18 tests (tiny PDFs built in-test, no extra dependency).

### Changed — Chunker CLI
- `python -m citerag.chunker` now loads files through `load_paths`, so it accepts PDFs and folders.

## 2026-10-25

### Added — `ai_agents/` build log
- Created this folder to record every step and the reasoning behind it.

### Added — Phase 2: Chunking ([02-chunking.md](02-chunking.md))
- `citerag/chunker.py`: structure-aware sentence packing with exact token counts, heading-aware sections, overlap on mid-paragraph cuts only, page-span tracking, stable chunk IDs, preview CLI.
- `tests/test_chunker.py`: 14 tests.
- Underlined headings (`Title` + `====`) detected; underlines removed from `embed_text`.

### Changed — Embedding model and chunk settings
- `EMBED_MODEL`: `all-MiniLM-L6-v2` → `BAAI/bge-small-en-v1.5` (MiniLM truncates at 256 tokens).
- Added `EMBED_MAX_TOKENS=512`, `EMBED_QUERY_PREFIX`, `CHUNK_MIN=96`.
- `CHUNK_SIZE` 500 → 384, `CHUNK_OVERLAP` 100 → 64.

### Added — Phase 1 (partial) ([01-loaders.md](01-loaders.md))
- `Page` dataclass in `citerag/loaders.py`.

### System — Python environment ([00-setup.md](00-setup.md))
- Installed `uv` 0.13.5 to `~/.local/bin`.
- Installed Python 3.12.15 via uv; created `.venv`; installed requirements.

### Added — Phase 0: Setup ([00-setup.md](00-setup.md))
- Project structure, `config.py`, `requirements.txt`, `.env.example`, `.gitignore`, `README.md`, placeholder `app.py`, `tests/test_setup.py`.
- `git init` (no commits yet).
