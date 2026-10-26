# Phase 3 — Embeddings + ChromaDB

**Status:** ✅ Done (2026-10-26) · Code: `citerag/store.py` · Tests: `tests/test_store.py` (12 passing)

## Goal

Turn every chunk into a vector (a list of numbers that captures its meaning) and save it in a persistent ChromaDB index, along with everything a citation needs. Retrieval quality is capped by the embedding model, so the model was picked by measurement, not by reputation.

## Step 1 — Picking the embedding model by benchmark

Five open models that run locally were tested on two standard retrieval benchmarks with human-labelled answers (BEIR **SciFact**: 5,183 scientific abstracts / 300 questions; **NFCorpus**: 3,633 medical documents / 323 questions). Documents were capped at 512 tokens, the same limit our chunks use. Hardware: Apple M5, 16 GB, MPS (GPU).

| Model | Size | Dims | SciFact nDCG@10 | SciFact Recall@5 | NFCorpus nDCG@10 | Speed (docs/s) |
|---|---|---|---|---|---|---|
| bge-small-en-v1.5 (old default) | 33M | 384 | 0.713 | 0.759 | 0.343 | ~130 |
| **bge-base-en-v1.5 (chosen)** | 110M | 768 | 0.740 | **0.815** | **0.374** | ~42 |
| snowflake-arctic-embed-m-v1.5 | 110M | 768 | 0.716 | 0.782 | 0.362 | ~41 |
| gte-modernbert-base | 149M | 768 | **0.751** | 0.799 | 0.347 | ~53 |
| Qwen3-Embedding-0.6B | 600M | 1024 | — | — | — | ~6 (fp32) / ~15 (fp16) on a small test |

- **nDCG@10** measures how high the right documents rank in the top 10. **Recall@5** is the share of right documents found in the top 5, which matters most because we pass `TOP_K = 5` chunks to Claude.
- The bge-small scores match its published numbers (0.713 / 0.343), which confirms the benchmark script is measuring correctly.

**Why bge-base:**
1. It has the best average across both datasets, the best Recall@5, and wins clearly on the harder medical set. gte-modernbert wins SciFact by 0.011 but loses NFCorpus by 0.027.
2. It is a **drop-in replacement**: it uses the same tokenizer (verified identical vocabulary), the same 512-token limit and the same query prefix as bge-small. The chunker's token budgets stay exactly right with no other changes.
3. Indexing speed is fine: 1,433 chunks took about 40 seconds, model load included.

**Why not Qwen3-Embedding-0.6B:** it ranks higher on public leaderboards, but on this Mac it embedded about 6 docs/s in fp32 (15 in fp16 on a 64-doc test). The full benchmark run had not finished the first dataset after 30+ minutes, so it was stopped. Re-indexing a document folder would take minutes instead of seconds. It can be switched in later through config if needed (set `CITERAG_EMBED_MODEL`, set `CITERAG_EMBED_QUERY_PREFIX` to its instruction, then re-index with `--reset`).

**Not considered:** models that need `trust_remote_code` (e.g. nomic-embed), which run arbitrary code from the model repo; gated models (embeddinggemma); paid APIs (e.g. Voyage), which would send documents to a third party and cost money per re-index.

## Step 2 — How chunks are embedded (`Embedder`)

| Choice | Why |
|---|---|
| Embed `chunk.embed_text` (the "title > section" header + normalized text), not the raw text | The vector also knows which document and section the passage belongs to, which helps when the passage itself never names its topic |
| Query prefix `"Represent this sentence for searching relevant passages: "` on **queries only** | bge was trained this way (asymmetric retrieval: short questions vs. long passages). Documents get no prefix |
| `normalize_embeddings=True` → unit vectors | Cosine similarity becomes a plain dot product; scores are comparable across queries (needed for `MIN_SCORE` in Phase 4) |
| `max_seq_length` capped at `EMBED_MAX_TOKENS` (512) | Nothing is silently truncated; the chunker already sizes chunks to fit |
| Device auto-select: CUDA → MPS (Apple GPU) → CPU | Uses the GPU on this Mac; override with `CITERAG_EMBED_DEVICE` |
| Model loaded lazily | Opening the store (e.g. `--stats`) doesn't pay the model load time |

## Step 3 — How vectors are stored (`Store`)

- **ChromaDB `PersistentClient`** at `.chroma/` (git-ignored), with telemetry off.
- **Cosine space**, plus a larger HNSW graph than Chroma's default (`ef_construction` 200, `ef_search` 128, `max_neighbors` 32). *Why:* the defaults favour speed. At our scale the bigger graph costs almost nothing and keeps the approximate search effectively exact, so no relevant chunk is missed by the index itself.
- **Chroma's built-in embedding model is disabled** (`embedding_function=None`). We always pass vectors ourselves, so Chroma can never quietly embed something with a different model.
- **Per chunk we store:** the vector, the **exact chunk text** (for showing and checking citations in Phase 6), and metadata: `source`, `page_start`, `page_end`, `section`, `start_char`, `end_char`, `n_tokens`, `fingerprint`. Chroma can't store `None`, so `section` is left out when there isn't one and read back as `None`.

## Step 4 — Safe, incremental re-indexing

- **Fingerprint per file:** a hash of the file's page texts **plus** the model name, document prefix and chunk settings. Re-running ingestion:
  - skips a file when the fingerprint is unchanged (no re-embedding, so it's fast);
  - otherwise deletes **all** of that file's old chunks and adds the new ones. Chunk IDs depend on position, so a partial update could leave stale chunks behind. Old chunks are deleted only *after* the new embeddings are computed, so a crash keeps the old version.
  - Changing `CHUNK_SIZE` etc. automatically re-chunks and re-embeds everything on the next run.
- **`--prune`** removes indexed files that are no longer in the folder.
- **Model lock:** the collection records which model built it. Opening it with a different model raises `EmbeddingMismatchError`. *Why:* vectors from two models live in different "spaces", and mixing them returns nonsense without any error. `Store(reset=True)` / `--reset` drops the index first, which is the supported way to switch models. (The first version reset *after* opening, so the suggested fix itself raised the error; caught in manual testing and covered by a test now.)

## CLI

```bash
python -m citerag.store data/              # index a folder (incremental)
python -m citerag.store data/ --prune      # ...and drop files no longer there
python -m citerag.store data/ --reset      # rebuild from scratch (e.g. after a model change)
python -m citerag.store --stats            # chunks per file + model
python -m citerag.store --query "..." -k 5 # nearest chunks with raw cosine scores
```

## Real-data check

Indexed two Apple GarageBand PDFs (1,433 chunks):
- *"Can I install the software on more than one computer?"* → top hit (0.654) was the exact licence clause about copies per Mac, pages 1–2. ✅
- *"What happens to the license if I break its terms?"* → the termination clause was **not** in the top 2. Plain vector search matches meaning loosely and can miss exact-wording answers. This is the motivation for Phase 4 (hybrid keyword + vector search and reranking).
- The 132-page agreement contains many languages; a Danish chunk scored close to the English one. Language filtering is a possible Phase 4 refinement.

## Tests (`tests/test_store.py`)

A deterministic fake embedder (hashed bag of words) keeps the tests offline and fast. They cover:
- query returns the closest chunk with all citation fields, and a missing section comes back as `None`
- unchanged files embed nothing; changed files leave no stale chunks; changed chunk settings trigger re-embedding
- prune, delete, reset; persistence across instances; source filtering
- model mismatch refused, and `reset=True` allows switching models
- the `Embedder` adds the prefix to queries only and returns normalized float32 vectors

## Known limitations
- Dense-only search; keyword matching and reranking come in Phase 4.
- `--query` is a raw debugging view, not the retriever: no score threshold, no deduplication.
- Re-indexing is per file: one edited page re-embeds the whole file (fast enough at this scale).
