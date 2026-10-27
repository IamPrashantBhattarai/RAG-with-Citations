# CiteRAG — Build Log

A record of every step taken to build this project: **what** was done and **why** it was done that way. Updated after every code change or system/environment change.

## Files

| File | Covers | Status |
|---|---|---|
| [00-setup.md](00-setup.md) | Project structure, Python environment, dependencies, git | ✅ Done |
| [01-loaders.md](01-loaders.md) | Loading PDF/TXT/MD into page records, text cleanup | ✅ Done |
| [02-chunking.md](02-chunking.md) | Splitting pages into chunks for embedding | ✅ Done |
| [03-vector-store.md](03-vector-store.md) | Embedding model benchmark, embeddings, ChromaDB index | ✅ Done |
| [04-retrieval.md](04-retrieval.md) | Hybrid search (dense + BM25), fusion, reranker benchmark, relevance gate | ✅ Done |
| [05-generation.md](05-generation.md) | Cited answers: Claude (native Citations) or free Gemini ([n] markers + quote location) | ✅ Done |
| [06-citation-checks.md](06-citation-checks.md) | Checking each statement against its citations (uncited claims, numbers/names, local NLI model), strict mode | ✅ Done |
| [07-ui.md](07-ui.md) | Streamlit UI: documents sidebar, streamed answers, sources with highlighted quotes and PDF pages, checks; safety and localhost-only | ✅ Done |
| [CHANGELOG.md](CHANGELOG.md) | Dated list of every change | Ongoing |

Upcoming: `08-evaluation.md`.

## Roadmap at a glance

| Phase | Goal | Status |
|---|---|---|
| 0 | Setup | ✅ |
| 1 | Document loaders | ✅ |
| 2 | Chunking | ✅ |
| 3 | Embeddings + ChromaDB | ✅ |
| 4 | Retrieval | ✅ |
| 5 | Cited answers (Claude or free Gemini) | ✅ |
| 6 | Citation validation | ✅ |
| 7 | Streamlit UI | ✅ |
| 8 | Evaluation + tuning | ⬜ |

## Key decisions so far

1. **Python 3.12 via `uv`**, not system Python 3.9 — 3.9 is end-of-life and too old for current ChromaDB/Streamlit. ([00-setup.md](00-setup.md))
2. **Embedding model `BAAI/bge-base-en-v1.5`**: first `bge-small` replaced `all-MiniLM-L6-v2` (MiniLM truncates at 256 tokens), then `bge-base` won a 5-model benchmark (best average nDCG@10 and Recall@5, same tokenizer as bge-small). ([02-chunking.md](02-chunking.md), [03-vector-store.md](03-vector-store.md))
3. **Structure-aware sentence packing** for chunking, 384 tokens / 64 overlap, measured in the embedding model's own tokens. ([02-chunking.md](02-chunking.md))
4. **Physical PDF page numbers + PDF cleanup** (headers/footers, hyphenation, inferred paragraph breaks) so citations match the viewer and the chunker can see headings. ([01-loaders.md](01-loaders.md))
5. **ChromaDB, cosine, our own vectors only**, with per-file fingerprints for incremental re-indexing and a model lock that refuses to mix vectors from different models. ([03-vector-store.md](03-vector-store.md))
6. **Hybrid search (dense + BM25, RRF) with no reranker by default**: every reranker tested was slower and not better on both benchmarks; the reranker stays as an option. Questions whose best match is below cosine 0.45 return nothing. ([04-retrieval.md](04-retrieval.md))
7. **Claude's native Citations feature, one plain-text document per chunk**, so every citation carries an API-extracted exact quote with character offsets; model `claude-opus-5-5` at effort `medium`, with server-side refusal fallback. ([05-generation.md](05-generation.md))
8. **Free Gemini backend** (`gemini-3.6-flash`; 3.8 was overloaded on the free tier) behind `CITERAG_LLM_PROVIDER`, for testing without a paid key: `[n]` markers are parsed while streaming and mapped to exact quotes, so both backends return the same cited `Answer`. ([05-generation.md](05-generation.md))
9. **Every answer is checked statement by statement** with a number/name check plus a local NLI model (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, threshold 0.5), chosen from 4 models on RAGTruth, hand-labelled syllabus claims and real Gemini answers; the faster tasksource model failed on tables. Strict mode removes unsupported statements. ([06-citation-checks.md](06-citation-checks.md))
10. **A plain Streamlit page, reachable only from this computer.** Every citation opens to its passage (quote highlighted) and the cited PDF page; unsupported statements are highlighted. LLM and document text is HTML-escaped before rendering, and `server.address = "localhost"` replaces Streamlit's listen-on-every-interface default. ([07-ui.md](07-ui.md))
