# Changelog

Newest first. Each entry links to the phase doc with the full reasoning.

## 2026-10-26

### System — Pushed to GitHub
- Committed the free Gemini backend, the Phase 5 live-test fixes and the build log to `main`. Before pushing: no Anthropic (`sk-ant-…`) or Google (`AIza…`) keys in tracked files; `.env` (which now holds the Gemini key) confirmed git-ignored.

### Done — Phase 5 live-tested with free Gemini ([05-generation.md](05-generation.md#step-7--live-test-users-syllabus-free-gemini-key))
- 4/4 test questions correct, with precise citations; the off-topic question made no API call.
- Default Gemini model `gemini-3.8-flash` → `gemini-3.6-flash` (3.8 returned 503 "high demand" to every free-tier request); fallback `gemini-3.5-flash-lite`.
- Fallback to the next model on 503/429/timeouts, only before any text is shown; mid-answer failures say so. SDK retries bounded (2 attempts, waits ≤ 8 s) plus a 30 s request timeout, replacing a silent >5-minute retry loop.
- Fixed: the CLI didn't catch backend errors under `python -m` (module loaded twice → two `GenerationError` classes).
- Fixed: tables produced one huge shared quote. `locate_quote` now considers lines inside long "sentences", picks the shortest near-best span, and trims trailing section numbers.
- Tests: 104 total.

### Added — Free Gemini backend for Phase 5 ([05-generation.md](05-generation.md#step-6--a-free-backend-google-gemini-added-2026-10-26))
- `citerag/gemini.py`: `GeminiBackend` (`gemini-3.8-flash`, free tier, thinking `low`, streaming, blocked-answer handling, plain messages for missing/rejected key, 404, 429 free-tier limit, 5xx, network).
- `citerag/citations.py`: `InlineCitations` parses `[n]` markers while the answer streams, renumbers them, and finds the exact supporting sentence (`locate_quote`, with `Citation.support`); invalid markers are dropped and recorded.
- `generator.py` refactored into `Generator` + backends (`ClaudeBackend`, `GeminiBackend`) behind `CITERAG_LLM_PROVIDER` / `--provider`; Claude behaviour unchanged.
- Config: `LLM_PROVIDER` (default `anthropic`), `GEMINI_MODEL`, `GEMINI_THINKING`. The user's `.env` now sets `CITERAG_LLM_PROVIDER=gemini` plus a `GEMINI_API_KEY` placeholder.
- Tests: `test_citations.py` (13), `test_gemini.py` (11); 96 total.

### System
- Installed `google-genai` 2.33.0 (added to requirements.txt). `tenacity` 9.2.1 → 9.1.4; dependency check clean.

### System — Pushed to GitHub
- Committed the loader fixes and Phases 4–5 to `main` and pushed to `IamPrashantBhattarai/RAG-with-Citations`. Before pushing: no API keys in tracked files; `.env`, `.venv/`, `.chroma/` and `data/*` (including the user's syllabus PDF) confirmed git-ignored.

### Added — Phase 5: Cited answers with Claude ([05-generation.md](05-generation.md)), live test pending
- `citerag/generator.py`: `Generator.answer(question, k, sources, on_text)` → `Answer` (text with [n] markers, `Citation`s with exact quotes and chunk/document offsets, `Segment`s including uncited text, `found`, usage).
- Each chunk is sent as a plain-text `document` with `citations: {enabled: true}`; the question comes last. The system prompt says: grounded only, say when not found, treat documents as data.
- Streaming via `client.beta.messages.stream` with live [n] markers; `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`); refusal and max_tokens handled.
- No API call when retrieval returns nothing.
- `GenerationError` with plain messages for missing/rejected key, 403/404/429/400/5xx and network errors.
- CLI: `python -m citerag.generator "question" [--source FILE] [-k N]`.
- `tests/test_generator.py`: 13 tests with a fake client (72 total).

### Changed — LLM config
- `LLM_MODEL`: `claude-sonnet-5-5` → `claude-opus-5-5`; `MAX_TOKENS`: 1024 → 16000 (it also covers thinking tokens); new `LLM_EFFORT=medium`.

### Added — Phase 4: Retrieval ([04-retrieval.md](04-retrieval.md))
- `citerag/retriever.py`: in-house BM25 keyword search (tokenizer with stopwords and plural folding), Reciprocal Rank Fusion, optional cross-encoder `Reranker`, `Retriever.retrieve(query, k, sources)`, and `passage()` (title > section + text).
- Relevance gate: returns `[]` when the best dense match is below `MIN_SCORE` (0.45, calibrated on on-topic vs. off-topic questions).
- CLI: `python -m citerag.retriever "question" [--rerank] [--source FILE] [-k N]`.
- `Store.get()`, `Store.all_chunks()`, `Store.version`, `Hit.scores`.
- `tests/test_retriever.py`: 13 tests (59 total).

### Decided — No reranker by default (benchmark on SciFact + NFCorpus)
- Hybrid 0.739 / 0.375 nDCG@10. Best reranker (gte-reranker-modernbert-base) 0.755 / 0.347 at about 1.5 s per question; MiniLM and bge-reranker-base were worse on SciFact.
- New config: `RETRIEVE_CANDIDATES=30`, `USE_RERANKER=0`, `RERANK_MODEL=Alibaba-NLP/gte-reranker-modernbert-base`. `MIN_SCORE` is now a cosine gate, 0.3 → 0.45.

### Fixed
- Retriever CLI ignored the config and always reranked (now `--rerank/--no-rerank`).
- CLIs run with `python -m` hid their own INFO messages (logger `__main__`); this affected the store's per-file lines since Phase 3.

### System
- Downloaded reranker models for the benchmark: ms-marco-MiniLM-L-6-v2, bge-reranker-base (1.1 GB), gte-reranker-modernbert-base. None are loaded by default.

### Fixed — Loader issues found on the first real PDF ([01-loaders.md](01-loaders.md#fixes-after-the-first-real-document-2026-10-26))
- Symbol/Wingdings bullet glyphs (U+F0B7 etc.) mapped to `•`.
- Paragraph inference no longer splits off a line that follows one ending in a connecting word or comma. This stopped table cells like `HTML5` / `XHTML` from becoming false ALL-CAPS section headings.
- Character table rewritten with ASCII escape codes after two literal invisible characters were silently lost (import crash).
- 2 new tests (46 total). Re-indexed `data/`: the syllabus was detected as changed and re-embedded automatically.

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
