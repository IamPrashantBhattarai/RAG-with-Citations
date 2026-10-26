# Phase 0 — Project Setup

**Status:** ✅ Done (2026-10-25)

## Steps

### 1. Created the project layout

```
citerag/          # the library: one module per pipeline stage
  config.py       # every tunable setting in one place
  loaders.py  chunker.py  store.py  retriever.py  generator.py  citations.py
app.py            # Streamlit UI (placeholder until Phase 7)
data/             # user documents (git-ignored)
tests/            # pytest
eval/             # evaluation questions + scoring (Phase 8)
ai_agents/        # this build log
```

**Why:** each module maps to exactly one RAG stage (load → chunk → store → retrieve → generate → verify). That makes each stage testable on its own and lets us swap one (e.g. a different vector DB) without touching the others. The UI is kept separate from the library so the same pipeline can later be served by FastAPI or a CLI.

### 2. Centralised settings in `citerag/config.py`

Every value (model names, chunk size, top-k, score threshold, DB path) is read from environment variables with a sensible default, loaded from `.env` via `python-dotenv`.

**Why:** tuning in Phase 8 means changing these numbers many times. Keeping them in one file — overridable without code edits — makes experiments fast and reproducible.

### 3. Dependencies (`requirements.txt`)

| Package | Role |
|---|---|
| `anthropic` | Claude API for answer generation |
| `chromadb` | Local, persistent vector database |
| `sentence-transformers` | Local embedding models (no API cost) |
| `pypdf` | PDF text extraction, page by page |
| `streamlit` | Web UI with minimal code |
| `python-dotenv` | Load `.env` secrets/settings |
| `pytest` | Tests |

**Why these:** everything except the LLM runs locally and free. ChromaDB needs no server. `pypdf` gives text per page, which is required for page-level citations.

### 4. Secrets and git hygiene

- `.env.example` committed as a template; real `.env` (with `ANTHROPIC_API_KEY`) is git-ignored.
- `.gitignore` also excludes `.venv/`, `.chroma/` (the vector DB), and `data/*` (user documents).
- Hosted on GitHub: `IamPrashantBhattarai/RAG-with-Citations`, branch `main`. Before each push, the files to be committed are scanned for API keys (`sk-ant-…`), and `.env`, `.chroma/` and `data/*` are confirmed ignored.
- `.vscode/settings.json` is committed with one setting (`python.terminal.useEnvFile`) so VS Code terminals load `.env`.

**Why:** API keys and private documents must never end up in git history.

### 5. Python environment

- The Mac only had system **Python 3.9.6** (end-of-life Oct 2025; current ChromaDB/Streamlit need newer).
- Installed **`uv`** into `~/.local/bin` (no admin rights needed), used it to install **Python 3.12.15** and create `.venv`, then installed requirements.
- Installed versions: anthropic 2.3.0, chromadb 1.5.9, streamlit 1.66.1, sentence-transformers 6.2.0.

**Why `uv`:** no Homebrew/pyenv was present. `uv` installs an isolated Python without touching the system one, and installs packages much faster than pip.

### 6. Smoke test — `tests/test_setup.py`

Checks that the core packages import and that config values are sane (`TOP_K > 0`, `overlap < chunk size`).

**Why:** confirms the environment works before any real code is written.

## How to reproduce

```bash
source ~/.local/bin/env          # put uv on PATH
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env             # add ANTHROPIC_API_KEY
.venv/bin/python -m pytest
```

## Notes

- A torch warning `Redirects are currently not supported in MacOs` appears when ML libraries load — harmless.
