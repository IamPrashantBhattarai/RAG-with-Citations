# CiteRAG

Ask questions about your own documents and get answers where every statement cites its source: the file, the page, and the exact passage. Each statement is then checked against what it cites, and anything the sources don't support is highlighted.

Add PDFs or notes, ask a question, and click any citation to see the quoted passage and the PDF page it came from. If the documents don't cover the question, it says so instead of guessing.

- Answers from Claude (paid) or Google Gemini (free tier)
- Search, embeddings and citation checks run locally on your machine
- A plain web page, reachable only from your own computer
- Tested on real documents: 53 of 55 questions answered correctly, and no made-up facts in 124 answers (see [Results](#results))

## How it works

```
documents ─► load ─► chunk ─► embed + store
                                    │
question ─► search (meaning + keywords) ─► rerank ─► top 5 passages
                                                           │
                                     LLM answer with [n] citations
                                                           │
                         check each statement against what it cites
```

| Stage | What happens | Code |
|---|---|---|
| Load | PDF, TXT and Markdown files are read page by page. PDF text is cleaned: repeated headers and footers are removed, hyphenated words are rejoined, and paragraphs are rebuilt. Page numbers match what a PDF viewer shows. | `citerag/loaders.py` |
| Chunk | Pages are split into passages of about 384 tokens, along sentence and section boundaries, with a small overlap so a fact isn't cut in half. | `citerag/chunker.py` |
| Embed and store | Each passage is turned into a vector with `BAAI/bge-base-en-v1.5` and saved in a local ChromaDB index. Re-indexing only re-embeds files that changed. | `citerag/store.py` |
| Search | Two searches find candidates. One matches meaning (vectors) and the other matches exact words, names and codes (BM25). Their results are merged, and a reranker model (`gte-reranker-modernbert-base`) reorders the top 10. If nothing is close enough to the question, no LLM is called. | `citerag/retriever.py` |
| Answer | The 5 best passages go to the LLM. Claude returns exact quotes through its Citations API. Gemini writes `[n]` markers, which are matched back to the supporting sentence. | `citerag/generator.py`, `gemini.py`, `citations.py` |
| Check | Each statement is checked three ways: does it have a citation, do its numbers and names appear in the cited passage, and does a local NLI model judge that the passage supports it. Unsupported statements are highlighted; strict mode removes them. | `citerag/validator.py` |
| Show | A Streamlit page with document upload, streamed answers, clickable sources, the cited PDF page and a list of checks. | `app.py`, `citerag/ui.py` |

The models and settings were chosen, or later confirmed, by benchmarks on public datasets and on real documents. The reasons, numbers and rejected alternatives are written up per phase in [`ai_agents/`](ai_agents/README.md).

## Requirements

- Python 3.12 (the version this was built and tested with)
- About 1.4 GB of disk space for the local models. They download automatically the first time they are used: embeddings 419 MB, reranker 574 MB, NLI checker 364 MB.
- An API key for the LLM, either one of:
  - **Gemini** (free): get a key at https://aistudio.google.com/apikey. On the free tier, Google may use your prompts to improve its products, so don't use it with private documents. `gemini-3.6-flash` allows 20 requests a day; after that, the app switches to `gemini-3.5-flash-lite` on its own.
  - **Claude** (paid): an Anthropic API key. It gives exact quotes chosen by the model itself.

It runs on CPU, on an NVIDIA GPU (CUDA) or on an Apple Silicon GPU (MPS), whichever is available.

## Setup

With [uv](https://docs.astral.sh/uv/) (recommended, fast):

```bash
git clone https://github.com/IamPrashantBhattarai/RAG-with-Citations.git
cd RAG-with-Citations
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env
```

Or with plain pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then open `.env` and add your key:

```bash
# Free Gemini
CITERAG_LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key

# or Claude
ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is git-ignored, so the key never ends up in the repository.

## Using the web app

```bash
.venv/bin/streamlit run app.py
```

It opens at http://localhost:8501. The first start takes a little longer while the models load.

1. **Add documents.** Upload PDF, TXT or MD files in the sidebar, or put them in `data/` and click **Rescan data/ folder**.
2. **Ask a question** in the box at the bottom.
3. **Read the answer.** Citations appear as small numbers. Below the answer:
   - **Sources**: one per citation. Open it to see the whole passage with the quote highlighted, and **Show PDF** to see the cited page.
   - **Checks**: every statement with ✓ or ✗ and the reason.
   - **Retrieved passages**: everything the model was given, with search scores.

Statements that fail the checks have a yellow background, and hovering shows why. The sidebar also has:
- **Answer with:** choose Gemini or Claude.
- **Search in:** limit the search to some documents.
- **Passages:** how many passages the model gets.
- **Strict mode:** removes unsupported statements instead of highlighting them.

The page only listens on `localhost`, so other devices on your network can't reach it.

## Using the command line

Every stage can also be run on its own, with the virtual environment active (`source .venv/bin/activate`):

```bash
python -m citerag.store data/                   # index a folder (only new or changed files are embedded)
python -m citerag.store data/ --prune           # ...and drop files that are no longer there
python -m citerag.store --stats                 # list indexed files and chunk counts

python -m citerag.retriever "What are the pass marks?"           # search only, no LLM
python -m citerag.generator "What are the pass marks?"           # cited answer + checks
python -m citerag.generator "..." --provider gemini --strict     # Gemini, unsupported statements removed
python -m citerag.generator "..." --source notes.pdf             # search one file only

python -m citerag.loaders data/paper.pdf        # see the text extracted from a file
python -m citerag.chunker data/paper.pdf --show 5               # see how it is split into chunks
```

## Settings

All settings live in `citerag/config.py` and can be overridden in `.env`. The ones you are most likely to change:

| Variable | Default | What it does |
|---|---|---|
| `CITERAG_LLM_PROVIDER` | `anthropic` | `gemini` or `anthropic` |
| `CITERAG_GEMINI_MODEL` | `gemini-3.6-flash` | Gemini model; `gemini-3.5-flash-lite` is the fallback |
| `CITERAG_LLM_MODEL` | `claude-opus-5-5` | Claude model |
| `CITERAG_TOP_K` | `5` | Passages given to the LLM |
| `CITERAG_USE_RERANKER` | `1` | `0` turns the reranker off (about 0.4 s faster per question, less accurate) |
| `CITERAG_MIN_SCORE` | `0.45` | Below this similarity, the question is treated as "not in the documents" |
| `CITERAG_VALIDATE` | `1` | `0` skips the citation checks |
| `CITERAG_STRICT` | `0` | `1` removes unsupported statements |
| `CITERAG_CHUNK_SIZE` | `384` | Tokens per chunk (changing it re-embeds everything on the next index) |
| `CITERAG_EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | Changing it needs `python -m citerag.store data/ --reset` |

`.env.example` lists the rest.

## Results

Phase 8 measured the whole system on 78 hand-written questions about three PDFs: a course syllabus, *Attention Is All You Need* and *Scaling Vision Transformers*. 55 questions have answers in the documents, 8 are on-topic but not answered there, and 15 are off-topic.

| | Result |
|---|---|
| Answerable questions answered correctly | 53 / 55 |
| Answers with a made-up fact | 0 of 124 |
| Unanswerable questions answered "not in the documents" | 8 / 8 |
| Off-topic questions stopped before calling the LLM | 15 / 15 |
| The right passage ranked first | 84% (56% without the reranker) |
| True statements wrongly flagged by the checks | 8% (28% before tuning) |
| Time per answered question | about 2.5 s (Gemini 3.5 Flash-Lite, Apple M5) |

Run the evaluation yourself (the three PDFs listed in `eval/questions.json` must be in `data/`):

```bash
python -m eval.retrieval                 # search quality and the relevance cutoff; local, about 1 minute
python -m eval.answers --tag mytag       # the full pipeline with the LLM; resumable if a rate limit stops it
python -m eval.answers --tag mytag --recheck    # re-run only the citation checks on saved answers
```

Details and every table: [`ai_agents/08-evaluation.md`](ai_agents/08-evaluation.md).

## Tests

```bash
.venv/bin/python -m pytest
```

191 tests, offline. Fake models stand in for the LLM, the embedding model and the NLI model, so no API key or download is needed. The UI tests run the real Streamlit app headless.

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `citerag/config.py` | All settings, overridable through `.env` |
| `citerag/loaders.py` | Read PDF/TXT/MD into pages, clean PDF text |
| `citerag/chunker.py` | Split pages into chunks with stable IDs |
| `citerag/store.py` | Embeddings and the ChromaDB index |
| `citerag/retriever.py` | Hybrid search, reranking, the "not in the documents" cutoff |
| `citerag/generator.py` | Cited answers; the Claude backend |
| `citerag/gemini.py` | The Gemini backend |
| `citerag/citations.py` | Turn `[n]` markers into exact quotes |
| `citerag/validator.py` | Check each statement against its citations; strict mode |
| `citerag/ui.py` | Safe rendering, quote highlighting, uploads, cited PDF pages |
| `eval/` | Question set and benchmarks (Phase 8) |
| `tests/` | pytest suite |
| `ai_agents/` | Build log: what was done in each phase and why |
| `data/` | Your documents (git-ignored) |
| `.chroma/` | The local index (git-ignored) |

## Privacy

- Your documents, the index and the models stay on your computer. Only the question and the 5 retrieved passages (with their file names, pages and section titles) are sent to the LLM provider you choose.
- On Gemini's free tier, Google may use prompts and answers to improve its products.
- LLM and document text is HTML-escaped before it is shown, so a document can't inject code into the page.

## Limitations

- **No OCR:** scanned PDFs without a text layer can't be read.
- **Tables and multi-column PDFs** come out as plain text, sometimes in the wrong order. Values at the end of table rows are the most common miss.
- **Questions spanning two documents** can end up with all 5 passages from one of them.
- **One question at a time:** follow-up questions don't see earlier answers.
- **The checks are a signal, not proof:** they catch most wrong numbers and names, but a swapped term that also appears in the cited passage can pass.

## Build log

The project was built in phases, each with its own write-up of what was done, why, and what was measured:

0. Setup · 1. Loaders · 2. Chunking · 3. Embeddings and index · 4. Retrieval · 5. Cited answers · 6. Citation checks · 7. Web app · 8. Evaluation and tuning

See [`ai_agents/README.md`](ai_agents/README.md) and the dated [`ai_agents/CHANGELOG.md`](ai_agents/CHANGELOG.md).
