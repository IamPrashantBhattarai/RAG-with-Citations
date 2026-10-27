# Phase 4 — Retrieval

**Status:** ✅ Done (2026-10-26) · Code: `citerag/retriever.py` · Tests: `tests/test_retriever.py` (13 passing)

## Goal

Given a question, return the few chunks most likely to contain the answer, best first. Return **nothing** when the documents don't cover the question, so the LLM in Phase 5 is never handed irrelevant text to "cite".

## The pipeline (default)

```
question ──► dense search (ChromaDB, bge-base, top 30)  ──┐
        └──► keyword search (BM25, top 30) ───────────────┤
                                                          ▼
            relevance gate: best dense match < 0.45 → return []
                                                          ▼
            Reciprocal Rank Fusion → top 5 chunks
```

Optional, off by default: a cross-encoder reranker reorders the fused candidates (`CITERAG_USE_RERANKER=1` or `--rerank`).

## Step 1 — Keyword search (BM25), written in-house

**Why add keyword search:** embeddings capture meaning but blur exact tokens: course codes (`CSIT.316`), error codes, names, rare technical terms. BM25 scores a chunk by how many of the question's words it contains, weighting rare words more (IDF), with diminishing returns for repeats (k1 = 1.2) and a penalty for long, padded chunks (b = 0.75).

**Why our own ~50 lines instead of a library:** the logic is small and worth seeing. It is checked against published numbers: our BM25 scores 0.674 nDCG@10 on SciFact and 0.327 on NFCorpus, in line with the standard BM25 baselines (about 0.67 and 0.33).

**Tokenizer:** lowercase alphanumeric words, English stopwords removed, plurals folded (`scripts→script`, `libraries→library`, `classes→class`). A unit test caught `classes→classe` in the first version, which would never have matched `class`; fixed with the `-sses/-ches/-xes` rule.

**What it indexes:** `passage(hit)` = `"title > section"` header + chunk text, the same context the embedding saw.

**Where it lives:** built in memory from all stored chunks on first use, and rebuilt automatically when the store changes (`store.version` and the chunk count are checked on every query). Fine up to tens of thousands of chunks.

## Step 2 — Fusion with Reciprocal Rank Fusion

Each list contributes `1 / (60 + rank)` per chunk. **Why RRF:** a cosine similarity (around 0.4–0.9) and a BM25 score (0–20+) are on different scales, and RRF uses only ranks, so no calibration is needed. A chunk ranked well by both searches beats one ranked first by only one of them. k = 60 comes from the original RRF paper.

## Step 3 — Benchmark: does hybrid search help, and does a reranker?

Same setup as Phase 3 (BEIR SciFact + NFCorpus, bge-base embeddings, Apple M5). Rerankers re-scored the top 30 candidates.

| Method | SciFact nDCG@10 | SciFact R@5 | NFCorpus nDCG@10 | NFCorpus R@5 | Time / question |
|---|---|---|---|---|---|
| Dense only | **0.740** | **0.815** | 0.374 | 0.142 | ~0.05 s |
| BM25 only | 0.674 | 0.733 | 0.327 | 0.123 | <0.01 s |
| **Hybrid (RRF)** — chosen | 0.739 | 0.805 | **0.375** | **0.143** | ~0.05 s |
| Hybrid + ms-marco-MiniLM-L-6-v2 | 0.701 | 0.758 | 0.360 | 0.139 | 0.15 s |
| Hybrid + bge-reranker-base | 0.729 | 0.769 | (stopped) | — | 2.2 s |
| Hybrid + gte-reranker-modernbert-base | **0.755** | 0.795 | 0.347 | 0.129 | 1.5 s |

**Findings**
- **Hybrid ≈ dense on these benchmarks.** Their questions are paraphrases with few exact codes or names, which is where keyword search shines. It costs nothing in accuracy or speed here, and on the real syllabus it fixed a ranking that dense search alone got wrong (see Step 6).
- **No reranker improved both datasets.** MiniLM (trained on web search queries) hurt both. bge-reranker-base was worse on SciFact and very slow, so it was stopped after the first dataset because its remaining runs couldn't change the verdict. gte was the best on SciFact (+0.016) but lost 0.028 on NFCorpus, and adds about 1.5 s per question.
- **Decision:** hybrid without a reranker as the default. The reranker code stays (tested, with gte as the model) behind `USE_RERANKER`, because these benchmarks are not the user's documents. Phase 8 will re-test it on real questions.
- **Changed in Phase 8 (2026-10-27):** on the user's documents the reranker won clearly (right passage first 0.56 → 0.84, answers 50 → 53/55 correct), so it is now **on by default**, reranking the top 10 candidates (`RETRIEVE_CANDIDATES` 30 → 10, as accurate and 3× faster). See [08-evaluation.md](08-evaluation.md).

## Step 4 — "Not in the documents" gate (`MIN_SCORE = 0.45`)

**Why a gate:** retrieval always returns *something*. Without a cut-off, "How do I bake bread?" would hand Claude five syllabus chunks.

**Why a query-level gate on the best dense cosine:** with no reranker, the dense cosine is the only score on an absolute scale (RRF and BM25 are relative). If even the single closest chunk is far away, nothing in the index is about the question.

**Calibration** (top-1 cosine with bge-base; 30 clearly off-topic questions vs. real questions):

| Collection | On-topic: lowest | On-topic: median | Off-topic: median | Off-topic: highest |
|---|---|---|---|---|
| User's syllabus (12 questions) | 0.465 | 0.629 | 0.361 | 0.435 |
| SciFact (300) | 0.502 | 0.728 | 0.479 | 0.754 |
| NFCorpus (323) | 0.428 | 0.685 | 0.483 | 0.686 |

At 0.45 the syllabus keeps **100%** of real questions and rejects **100%** of off-topic ones. On the medical benchmarks, off-topic scores overlap with on-topic ones, partly because some "off-topic" questions (common cold, back pain) really are medical. **The threshold is deliberately conservative:** wrongly rejecting a real question is worse than letting a borderline one through, because Phase 5 will tell Claude to answer only from the sources and Phase 6 checks the citations.

⚠️ The value depends on the embedding model. Re-calibrate if `EMBED_MODEL` changes.

## Step 5 — API and CLI

```python
from citerag.retriever import Retriever
hits = Retriever().retrieve("What are the pass marks?", k=5, sources=None)
# [] if not covered. Each Hit: text, source, page_start/page_end, section,
# start_char/end_char, score (ordering), scores {"dense", "keyword", "rrf", "rerank"}
```

```bash
python -m citerag.retriever "What are the pass marks?"      # hybrid (default)
python -m citerag.retriever "..." --rerank                   # try the reranker
python -m citerag.retriever "..." --source notes.pdf         # limit to a file (repeatable)
```

Store additions for this phase: `Store.get(ids)`, `Store.all_chunks()`, `Store.version`, and `Hit.scores`.

## Step 6 — Real-data check (user's syllabus PDF, 7 chunks)

| Question | Result (default hybrid) |
|---|---|
| "What topics are covered in the unit about server side scripting?" | ✅ #1 p3. **Dense alone ranked it #2** (0.581 vs. a JavaScript chunk at 0.608); the keyword match (BM25 5.2) lifted it to #1 |
| "Which course is CSIT.316?" | ✅ #1 p1, the course header (exact code via keyword search) |
| "What are the pass marks?" | ✅ #1 p3–4, the evaluation section |
| "How do I bake sourdough bread?" | ✅ nothing returned (best match below 0.45) |

## Bugs found and fixed during this phase
1. **CLI always reranked.** `--no-rerank` was the only switch, so the CLI ignored the config default. Replaced with `--rerank/--no-rerank`, defaulting to `USE_RERANKER`.
2. **Hidden progress messages with `python -m`.** A module run as a command logs under the name `__main__`, not `citerag.*`, so its INFO lines were filtered out. This also affected the store's per-file "added/updated" lines since Phase 3. Both CLIs now enable `__main__` too.
3. **Plural folding:** `classes→classe` (see Step 1).

## Tests (`tests/test_retriever.py`)
Tokenizer; BM25 rare-term preference, length normalisation and no-match behaviour; RRF agreement; exact-code lookup through keyword search with all stage scores; the reranker decides order but never drops hits; the gate returns `[]` for unrelated questions; order without the reranker; source filtering in both searches; the keyword index rebuilds after new documents; empty query and empty index; the passage header. Fake embedder and fake reranker, so the tests run offline.

## Known limitations
- The gate is per question, not per chunk. Once a question passes, all top-5 chunks are returned even if only one is relevant. Phase 5's prompt and Phase 6's checks handle the rest.
- Without a reranker, `hit.score` is an RRF value (about 0.016–0.033), which is only useful for ordering. Use `hit.scores["dense"]` for a similarity.
- The BM25 index lives in memory per process. A second process (e.g. the UI) picks up re-indexing done elsewhere when the chunk count changes. An update that keeps the same count is only seen after a restart.
- No query rewriting or multi-question decomposition yet.
