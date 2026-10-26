# Phase 2 — Chunking

**Status:** ✅ Done (2026-10-25) · Code: `citerag/chunker.py` · Tests: `tests/test_chunker.py` (14 passing)

## Goal

Split documents into pieces that (a) each carry one coherent idea so the embedding represents it well, (b) fit fully inside the embedding model, and (c) can be traced back to an exact file, page and passage for citations.

## Step 1 — Switched the embedding model

> **Update (Phase 3):** the default is now `bge-base-en-v1.5`, chosen by benchmark. It shares bge-small's tokenizer and 512-token limit, so everything below still applies. See [03-vector-store.md](03-vector-store.md).

| | `all-MiniLM-L6-v2` (old plan) | `BAAI/bge-small-en-v1.5` (chosen) |
|---|---|---|
| Max input the model actually reads | **256 tokens** | **512 tokens** |
| Retrieval quality | Good | Better (higher on MTEB retrieval benchmarks) |
| Size / speed | ~23M params | ~33M params, similar speed |

**Why:** the roadmap planned 500-token chunks, but MiniLM's `max_seq_length` is 256 — anything beyond is silently cut off, so half of each chunk would never be embedded. Verified by reading each model's `sentence_bert_config.json`.

**Side effect:** bge expects the prefix `"Represent this sentence for searching relevant passages: "` on **queries only** (not documents). Saved as `EMBED_QUERY_PREFIX` in config for Phase 4.

## Step 2 — Chose the chunking technique

Options considered:

| Technique | Verdict |
|---|---|
| Fixed character/token windows | ❌ Cuts mid-sentence; meaning gets split across chunks |
| Recursive character splitting | 🟡 Better, but counts characters not tokens, and ignores headings |
| Semantic chunking (embedding-based breakpoints) | ❌ Much more compute; published comparisons show no consistent win over sentence/recursive methods |
| **Structure-aware sentence packing** | ✅ Chosen |

**Why sentence packing:** a sentence is the smallest unit that holds a complete thought. Packing whole sentences, while respecting headings and paragraphs, keeps each chunk coherent at almost no compute cost.

## Step 3 — The algorithm

1. **Join pages of a document into one text**, remembering where each page starts.
   *Why:* paragraphs often continue across a page break. Chunking per page would split them. Each chunk records `page_start` and `page_end`. If a page ends mid-sentence, it is joined with a space (not a paragraph break) so the sentence stays whole.

2. **Split into paragraphs → detect headings → split into sentences.**
   - Headings: `# Markdown`, underlined (`Title` + `====`), numbered (`2.1 Results`, `Chapter 3`), ALL CAPS lines.
   - Sentence splitter handles abbreviations (`e.g.`, `Dr.`, `J. Smith`) and list items (`- item`, `1. item`).
   - A sentence longer than the chunk size is split into word windows.
   *Why:* these are the natural meaning boundaries in a document.

3. **Count tokens with the embedding model's own tokenizer.**
   *Why:* word or character counts only approximate tokens. Exact counts guarantee no chunk is truncated at embedding time.

4. **Greedily pack sentences into chunks** up to `CHUNK_SIZE`, with these break rules:
   - A heading starts a new chunk (if the current one has ≥ `CHUNK_MIN` tokens).
   - If the next paragraph won't fit, break cleanly before it.
   - Only if a single paragraph is too big is it cut mid-paragraph.
   *Why:* chunks align with how the author organised ideas.

5. **Overlap only on mid-paragraph cuts**, by repeating the last whole sentences (≤ `CHUNK_OVERLAP` tokens).
   *Why:* overlap exists to repair a thought split across two chunks. At a clean paragraph or section break there is nothing to repair, so overlap would only add duplicate text to the index. Overlap never crosses a heading.

6. **Merge a tiny final chunk** (< `CHUNK_MIN`) into the previous one.
   *Why:* a 1-sentence scrap makes a weak, noisy embedding.

7. **Build two texts per chunk:**
   - `text` — an **exact slice** of the source. *Why:* Phase 6 verifies that cited quotes really exist in the source.
   - `embed_text` — `"title > section"` header + whitespace-normalised text (heading underlines removed). *Why:* a passage like "It increased by 12%" is ambiguous alone. The header tells the embedding what document and section it belongs to, which improves retrieval.

8. **Stable IDs:** `{file-slug}-p{page}-c{index}` (e.g. `my-notes-txt-p1-c0`).
   *Why:* deterministic, so re-ingesting a file produces the same IDs (no duplicates in the vector DB), and readable when debugging.

## Step 4 — Chose the sizes

| Setting | Value | Reasoning |
|---|---|---|
| `CHUNK_SIZE` | **384** tokens | ≈ 2–3 paragraphs. Big enough for context, small enough that a citation points to a specific passage. Leaves room for the header within 512. |
| `CHUNK_OVERLAP` | **64** tokens | ≈ 1–2 sentences (~17%). Whole sentences only. |
| `CHUNK_MIN` | **96** tokens | Below this, embeddings carry too little meaning. |
| Hard limit | **462** | 512 − 2 special tokens − 48 reserved for the header. Larger sizes are rejected with an error. |

**Honest caveat:** these are well-reasoned defaults, not yet measured on our documents. Real tuning happens in Phase 8 with retrieval hit-rate on a question set.

Early check on Python's LICENSE file (~2,000 words):

| Size | Chunks | Median tokens | Max |
|---|---|---|---|
| 256 | 17 | 206 | 255 |
| 384 | 13 | 245 | 382 |

## Step 5 — Tests (`tests/test_chunker.py`)

Most tests use a word counter instead of the real tokenizer so they run offline and fast. One test uses the real bge tokenizer to prove `embed_text` always fits in 512 tokens.

Covered: size limits, exact source slices, every sentence covered, overlap behaviour, no overlap at paragraph breaks, Markdown and underlined headings → sections, tiny-tail merge, page-break spanning, over-long sentences, abbreviations/lists, stable unique IDs, invalid settings rejected.

## Step 6 — Preview CLI

```bash
python -m citerag.chunker notes.md report.pdf --size 384 --overlap 64 --show 3
```

Prints chunk count, min/median/max token sizes, and the first few chunks' `embed_text`. Used to eyeball settings before Phase 8's proper evaluation.

## Config changes made in this phase

```diff
- EMBED_MODEL = "all-MiniLM-L6-v2"
+ EMBED_MODEL = "BAAI/bge-small-en-v1.5"
+ EMBED_MAX_TOKENS = 512
+ EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
- CHUNK_SIZE = 500
+ CHUNK_SIZE = 384
- CHUNK_OVERLAP = 100
+ CHUNK_OVERLAP = 64
+ CHUNK_MIN = 96
```
