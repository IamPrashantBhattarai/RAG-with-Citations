# CiteRAG

[![Tests](https://github.com/IamPrashantBhattarai/RAG-with-Citations/actions/workflows/tests.yml/badge.svg)](https://github.com/IamPrashantBhattarai/RAG-with-Citations/actions/workflows/tests.yml)

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
| Load | PDF, Word (.docx), PowerPoint (.pptx), HTML, email (.eml, .msg, with their attachments), TXT, Markdown and image files are read page by page, and web pages can be saved by URL (slides one per page with their speaker notes, cited as "slide 3"; Word and text files as one text, cited by section). Scanned pages, photos and images are read with OCR (RapidOCR); a page that is mostly a form is read row by row, so each value stays next to its label. PDFs go through Docling's layout model: reading order across columns, headings, and tables written one row per line with each value next to its column name. Page headers, footers and text inside figures are dropped. Parsed files are cached, so only new or changed files are parsed again. If Docling can't read a file, pypdf does. Page numbers match what a PDF viewer shows. | `citerag/loaders.py`, `citerag/layout.py` |
| Chunk | Pages are split into passages of about 384 tokens, along sentence and section boundaries, with a small overlap so a fact isn't cut in half. | `citerag/chunker.py` |
| Embed and store | Each passage is turned into a vector with `BAAI/bge-base-en-v1.5` and saved in a local ChromaDB index. Re-indexing only re-embeds files that changed. | `citerag/store.py` |
| Search | Two searches find candidates. One matches meaning (vectors) and the other matches exact words, names and codes (BM25). Their results are merged, and a reranker model (`gte-reranker-modernbert-base`) reorders the top 10. If nothing is close enough to the question, no LLM is called. | `citerag/retriever.py` |
| Answer | The 5 best passages go to the LLM. Claude returns exact quotes through its Citations API. Gemini writes `[n]` markers, which are matched back to the supporting sentence. | `citerag/generator.py`, `gemini.py`, `citations.py` |
| Check | Each statement is checked three ways: does it have a citation, do its numbers and names appear in the cited passage, and does a local NLI model judge that the passage supports it. Unsupported statements are highlighted; strict mode removes them. | `citerag/validator.py` |
| Show | A Streamlit page with document upload, streamed answers, clickable sources, the cited PDF page and a list of checks. | `app.py`, `citerag/ui.py` |

The models and settings were chosen, or later confirmed, by benchmarks on public datasets and on real documents. The reasons, numbers and rejected alternatives are written up per phase in [`ai_agents/`](ai_agents/README.md).

## Requirements

- Python 3.12 (the version this was built and tested with)
- About 2.4 GB of disk space for the local models. They download automatically the first time they are used: embeddings 419 MB, reranker 574 MB, NLI checker 364 MB, PDF layout 328 MB and table structure 683 MB (Docling), OCR 31 MB (RapidOCR, installed with Docling). The optional formula model (`CITERAG_PDF_FORMULAS=latex`) adds 1.2 GB.
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

1. **Add documents.** Upload PDF, Word (.docx), PowerPoint (.pptx), HTML, email (.eml, .msg), TXT, MD or image files (PNG, JPG, TIFF…; scans and photos are read with OCR) in the sidebar, or put them in `data/` and click **Rescan data/ folder**. To add a web page, paste its address under **Add a web page**: it is saved into `data/` (public http(s) pages only, up to 5 MB) and indexed.
2. **Ask a question** in the box at the bottom. Follow-ups work ("And the deposit?"): a follow-up is rewritten into a standalone question, shown under it as *Searched for: …*, and answered from the documents like any question.
3. **Read the answer.** Citations appear as small numbers. Below the answer:
   - **Sources**: one per citation. Open it to see the whole passage with the quote highlighted, and **Show PDF** to see the cited page.
   - **Checks**: every statement with ✓ or ✗ and the reason.
   - **Retrieved passages**: everything the model was given, with search scores.

Statements that fail the checks have a yellow background, and hovering shows why. The sidebar also has:
- **Answer with:** choose Gemini or Claude.
- **Search in:** limit the search to some documents.
- **Passages:** how many passages the model gets.
- **Strict mode:** removes unsupported statements instead of highlighting them.
- **Chats:** every conversation is saved (in `.chats/`, on this computer) and listed here; *New chat* starts another. The address bar holds the chat's id, so a reload or bookmark reopens it.

The page only listens on `localhost`, so other devices on your network can't reach it.

## Sign-in and permissions (optional)

Without this, CiteRAG is a single-user tool on your computer. With it, people sign in with Google, only those listed in `access.toml` get in, and each one searches only the documents of their groups.

**How permissions work:** folders under `data/` are groups. `data/hr/…` is for the group `hr`; `data/public/…` is for everyone signed in; files directly in `data/` (in no folder) are for admins only. Admins see everything and are the only ones who can upload, add web pages, remove documents and rescan. Each person's chats are their own.

**Set it up (about 20 minutes).** You need a Google account. No billing account, and no APIs to turn on.

1. **Project.** In [Google Cloud Console](https://console.cloud.google.com/), create a project (*project picker → New project*) and select it.
2. **Sign-in screen.** Open *Google Auth Platform* (formerly *APIs & Services → OAuth consent screen*) and click *Get started*. Enter an app name and a support email, then choose the audience:
   - *Internal* if everyone who'll sign in has an account in your Google Workspace. This is only offered when the project belongs to that organisation.
   - Otherwise *External*. While the app is in *Testing*, add everyone who will sign in under *Audience → Test users*.

   Under *Data Access*, the scopes are `openid`, `email` and `profile`. Nothing else is needed.
3. **OAuth client.** Go to *Clients → Create client* and choose *Web application*.
   - Authorised redirect URI, exactly: `http://localhost:8501/oauth2callback` (`http`, `localhost`, port 8501, no trailing slash).
   - Leave *Use this client for an AI-powered agent* unticked. That option is for clients an AI agent uses to work with someone's Google data.
   - Copy the **Client ID** and **Client secret** straight away, because Google shows the secret in full only once.
4. **Secrets.** Open `.streamlit/secrets.toml` (created for you, not in git). Paste the two values over the `PASTE…` placeholders, keeping both quotes. Everything else is already filled in, including a random cookie secret.
5. **Access list.** Edit `access.toml` (not in git; see `access.example.toml`): who may sign in, their groups, and who is an admin. Changes apply on the next page load.
6. **Restart** the app and open `http://localhost:8501` (not `127.0.0.1`). To move existing documents into group folders, move the files under `data/` and click **Rescan**.

With *External*, two lists must agree:
- Google's test users decide who can get through Google's sign-in.
- `access.toml` decides who CiteRAG lets in and what they see.

Publishing the app removes the test-user list. Google doesn't review apps that ask only for these basic scopes, so after that `access.toml` is the only list.

**If sign-in doesn't work:**

| You see | Fix |
|---|---|
| The normal app, with no sign-in button | Both `PASTE…` placeholders are still in `secrets.toml`, or the app wasn't restarted |
| "The sign-in settings have a problem, so the app won't open" | `secrets.toml` isn't valid TOML (often a missing quote; the message gives the line), or only some settings are filled in. Fix it and restart. The app won't open without sign-in while the settings are broken. |
| `Error 400: redirect_uri_mismatch` | The URI in the Google client must equal `redirect_uri` in `secrets.toml`, and the app must be on port 8501. New client settings can take a few minutes to apply. |
| `Error 403: access_denied` | Add the account under *Audience → Test users* |
| `Error 403: org_internal` | The audience is *Internal*: use a Workspace account, or switch to *External* |
| `Error 401: invalid_client` | The Client ID or secret is mistyped, or belongs to another client |
| "… isn't on the access list" | Google signed you in, but that email isn't in `access.toml` |

The [Phase 13 build log](ai_agents/13-access.md) explains the design and the leak tests.

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
python -m citerag.web https://example.org/page  # save a web page into data/ (then index data/)
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
| `CITERAG_PDF_PARSER` | `docling` | `pypdf` reads PDFs without the layout model (faster, but tables and columns come out as run-on text) |
| `CITERAG_PDF_FORMULAS` | `text` | `latex` writes formulas as LaTeX where the formula model agrees with the PDF text (1.2 GB model, about 20 s per formula) |
| `CITERAG_EMAIL_ATTACHMENTS` | `1` | `0` lists an email's attachments by name instead of reading them |
| `CITERAG_OCR` | `rapidocr` | OCR for pages without a text layer and for images: `ocrmac` uses Apple Vision on a Mac (`pip install ocrmac`; fewer glued words on printed scans), `off` turns OCR off |
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

286 tests:
- **Fakes stand in for most models.** The LLM, sign-in and most of the search and checking models are faked, so no API key is needed. The UI tests run the real Streamlit app headless.
- **Some tests use real models.** The parsing and OCR tests run Docling and RapidOCR for real, one index test loads the embedding model, and one chunker test loads its tokenizer. They download about 500 MB on the first run and work offline after that.
- **Two tests read the evaluation documents in `data/`** and are skipped when those aren't there.

**On GitHub** ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)), the tests run on a clean Linux machine for every push and pull request; the badge at the top shows the latest result.
- The machine installs exactly the versions in [`requirements.lock`](requirements.lock), so it tests what runs locally rather than whatever is newest. To install those versions yourself: `uv pip install --python .venv/bin/python -r requirements.lock --torch-backend cpu`.
- No secrets are needed.
- The two document tests are skipped there, because `data/` isn't in git.

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `citerag/config.py` | All settings, overridable through `.env` |
| `citerag/loaders.py` | Read PDF/TXT/MD into pages, clean PDF text |
| `citerag/layout.py` | Docling parsing: PDFs (layout, OCR, forms), Word, PowerPoint, HTML; tables as rows, checked formulas, parse cache |
| `citerag/web.py` | Save a web page by URL into `data/`, public addresses only |
| `citerag/mail.py` | Pull the attachments out of an email, for the loader to read |
| `citerag/conversation.py` | Rewrite follow-up questions into standalone ones before searching |
| `citerag/chats.py` | Save and reopen chats (`.chats/`, one folder per user when sign-in is on) |
| `citerag/auth.py` | Google sign-in through Streamlit (which verified email is signed in) |
| `citerag/access.py` | Who may read which documents (`access.toml`), and the search that enforces it |
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
- LLM and document text is HTML-escaped before it is shown, so a document can't inject code into the page. Formulas (`$…$`, `$$…$$`) are rendered with KaTeX in untrusted mode.

## Limitations

- **Web pages:** the clean-up that strips menus and banners is a heuristic, and pages built by JavaScript or behind a login can't be read.
- **OCR** was tested on one handwritten form (17/17 values read) and on scanned copies of the papers (about 96% of words exact). RapidOCR sometimes drops spaces between words on printed scans, and only English was tested.
- **Forms:** the citation checker can still flag a correct form value whose label is on another row (the EIN on the test form). Table values are read off the table, so a number from the wrong column is flagged with the column it actually belongs to.
- **Tables:** each row is now one line naming its columns, but a question can still use different words from the table ("parameters" vs the header "params ×106"), and that row is still missed.
- **Formulas** keep the PDF's own text ("softmax( QKT √dk )V"). Docling's LaTeX model misread 3 of 5 formulas in the Attention paper, so it is off by default.
- **Questions spanning two documents** can end up with all 5 passages from one of them.
- **One question at a time:** follow-up questions don't see earlier answers.
- **The checks are a signal, not proof:** they catch most wrong numbers and names, but a swapped term that also appears in the cited passage can pass.

## Build log

The project was built in phases, each with its own write-up of what was done, why, and what was measured:

0. Setup · 1. Loaders · 2. Chunking · 3. Embeddings and index · 4. Retrieval · 5. Cited answers · 6. Citation checks · 7. Web app · 8. Evaluation and tuning · 9. Layout-aware PDF parsing

See [`ai_agents/README.md`](ai_agents/README.md) and the dated [`ai_agents/CHANGELOG.md`](ai_agents/CHANGELOG.md).
