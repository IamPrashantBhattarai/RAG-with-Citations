# Changelog

Newest first. Each entry links to the phase doc with the full reasoning.

## 2026-11-11

### Added — Phase 13: Sign-in and permissions ([13-access.md](13-access.md))
- **Google sign-in** through Streamlit's built-in OpenID Connect (`st.login`/`st.user`/`st.logout`), in `citerag/auth.py`. It's on only when `.streamlit/secrets.toml` `[auth]` is filled in; otherwise the app is the local single-user tool it was. Only Google-verified emails count.
- **Permissions** (`citerag/access.py`, `access.toml`):
  - only listed emails get in; folders under `data/` are groups; `public/` is for everyone signed in; files in no folder are admins only
  - admins read everything and are the only ones who upload, add web pages, remove and rescan
  - `ScopedRetriever` filters every search to the user's documents and checks every hit again
  - an empty set returns nothing (the Phase 4 retriever treats an empty filter as "all")
- **App:** sign-in screen, "Signed in as", per-user document list and "Search in", an upload folder (group) choice for admins, uploads indexed with their folder in the name, per-user chats (`.chats/<key>/`), and saved answers hidden when they used a document the user can no longer read. `access.toml` is re-read when edited.
- **Leak test** `eval/permissions.py` (new): 126 eval questions × 4 restricted users on the real search stack, 504 searches, 0 passages from documents a user may not read.
- **Fixed:** chats saved in the same second were listed in random order (second-resolution timestamps).
- **Manual step for the user:** create a Google OAuth client (redirect `http://localhost:8501/oauth2callback`), paste its id and secret into `.streamlit/secrets.toml`, edit `access.toml`. The README has the steps.
- Tests: `test_access.py` (14), +6 `test_ui.py`. 275 passing.

### System
- Installed `Authlib` 1.9.4 and `joserfc` 1.8.0 (`--exclude-newer 2026-11-04`). `requirements.txt`: `streamlit[pdf,auth]>=1.66`.
- New git-ignored files: `access.toml` (lists people's emails) and `.streamlit/secrets.toml` (already ignored; now created with a random cookie secret). `access.example.toml` committed.

### Added — Phase 12: Multi-turn chat ([12-conversations.md](12-conversations.md))
- `citerag/conversation.py` (new): a follow-up ("And how big is the deposit?") is rewritten by the LLM from the last three turns into a standalone question, which then goes through the unchanged search → cited answer → checks pipeline. The history never reaches the answering prompt. A cheap check (`is_follow_up`: pointing words, "what about…", four words or fewer) skips the call for questions that stand alone; on the 126 standalone eval questions it flags 3. Bad or failed rewrites fall back to the question as typed.
- `Backend.complete()` on both backends. Claude: `beta.messages.create`, effort `low` (Claude Opus 5.5 always thinks and defaults to `medium`), 4,096 tokens, the same refusal fallback. Gemini: thinking `LOW`, same next-model fallback. Anthropic error mapping moved into one shared `_anthropic_errors`.
- `citerag/chats.py` (new): one JSON file per chat in `.chats/` (git-ignored, `CITERAG_CHATS_DIR`), with each turn's question, what was searched, and the whole answer including its passages, so old chats keep their sources after a re-index. Ids are validated (they come from the URL).
- App: the chat id is in the address (reload or bookmark to reopen); the sidebar **Chats** section has *New chat*, recent chats and *Delete this chat*, replacing *Clear chat*; a rewritten question shows "Searched for: …".
- **Eval:** `eval/followups.json` and `eval/followups.py` (new): 12 two-turn chats built on real earlier answers, including a topic change. Evidence in the top 5: as typed 11/12, rewritten 12/12. **Answers: as typed 9/12, rewritten 12/12.** As typed, "How long does it run?" was answered about the Transformer's training time instead of the lease.
- Tests: `test_conversation.py` (5), `test_chats.py` (4), +1 `test_generator.py`, +2 `test_gemini.py`, +2 `test_ui.py`. 255 passing.

### Added — Phase 11 (part 4): Email ([11-more-formats.md](11-more-formats.md#email-eml-msg))
- `.eml` and Outlook `.msg` are read with Docling's email reader: the subject as title, From/To/Cc/Date, then the body. Cited by name and section.
- **Attachments are read** (`CITERAG_EMAIL_ATTACHMENTS`, on by default): `citerag/mail.py` (new) extracts them, and each is read by the same loader as an upload. It's added under "## Attachment: <name>", with its headings renamed "### <name>: …". Inline and small images (logos), files over 25 MB and attachments past the first 20 are skipped; unreadable ones are listed. Safe names, each in its own temporary folder.
- Forwarded emails: their text is already in Docling's body, so only their files are read (one level). The first version read the text twice.
- **Eval:** `data/pokhara-office-lease.eml` (a test email written for this) and 7 questions, two answered only by its Word attachment, plus un-13 (a phone number only in an unread CSV). Retrieval: 7/7 in the top 5. Answers: 7/7 correct, with no checker warnings; un-13 declined.
- Tests: `test_mail.py` (4, new). 241 passing.
- Phase 11 is done: Word, PowerPoint, HTML and web pages, and email.

### Added — Phase 11 (part 3): HTML files and web pages ([11-more-formats.md](11-more-formats.md#html-files-and-web-pages))
- `.html` / `.htm` files are read as one text, cited by name and section. Menus, banners and footers are left out, and so are link targets (for Word too).
- `layout.clean_html`: before Docling, a page is cut down to its `<main>` or `<article>`, without clutter tags, hidden elements, or elements whose id or class has a clutter word (`toc`, `menu`, `cookie`, `lang`, `editsection`, `reference`…, whole words only). On a real Wikipedia article: 56,566 → 29,301 characters, starting at the article, with no `[ 1 ]` markers the LLM could mistake for its own citations.
- `citerag/web.py` (new): save a page by URL into `data/`, with "Web page: <url>" at the top of its text. Only public http(s) addresses, checked at every redirect, so the app can't be pointed at localhost, the LAN or cloud metadata. Also capped: 5 redirects, 15 s, 5 MB, HTML or PDF only. App: "Add a web page" in the sidebar. CLI: `python -m citerag.web URL`.
- **Eval:** `data/rooftop-solar-guide.html` (a test page written for this) and 6 questions plus un-12. Retrieval: 6/6 in the top 5. Answers: 6/6 correct, with no checker warnings; un-12 declined.
- Tests: `test_web.py` (4, new), +2 `test_layout.py`. 237 passing.

### Added — Phase 11 (part 2): PowerPoint files ([11-more-formats.md](11-more-formats.md#powerpoint-pptx))
- `.pptx` files are read with Docling's PowerPoint reader, one page per slide. The slide title becomes the section, and tables become rows. **Speaker notes** are kept after their slide ("Speaker notes: …"). Citations say "slide 3". Old `.ppt` files get "save it as .pptx or PDF".
- `citerag/chunker.py`: **one chunk per slide**, and slides always start a new paragraph. Before, slides 1–4 shared a chunk: they were cited as "slides 1–4", slide titles weren't recognised as headings (pages without a full stop were joined with a space), and the speaker-notes question fell below the relevance gate (0.431). Now it scores 0.527.
- `CHUNKER_VERSION` (2) is part of the index fingerprint, so a change in chunking logic re-chunks every file once. The next index re-embeds all files.
- **Eval:** `data/q3-business-review.pptx` (a test deck written for this) and 9 questions plus un-11. Retrieval: 9/9 in the top 5. Answers: 9/9 correct, each citing the exact slide, with no checker warnings; un-11 and the off-topic exchange-rate question declined.
- Tests: +1 `test_layout.py`, +1 `test_loaders.py`, +1 `test_chunker.py`; `test_generator.py` extended. 231 passing.

### Changed — The checker reads tables ([06-citation-checks.md](06-citation-checks.md#update-2026-11-11-tables-and-forms))
- `citerag/validator.py`: when every number in a statement comes from table rows, `table_verdict` checks that each number sits in a column and row the statement names, with no NLI call. A number from another column or row is flagged precisely: "6,000 is the Daily allowance (NPR) for Grade A (Director) in the cited table". Unclear cases go to NLI as before. Form rows also get the two lines above them as a premise (`line_windows`).
- `eval/checks.py` (new): a re-runnable checker benchmark on saved answers. True statements, plus each number swapped for another from the same table row or line.
- Measured on 101 true and 73 altered statements: false alarms 10 → 8, altered caught 66 → 69, balanced accuracy 0.903 → 0.933. Fixed docx-06, docx-05, form-08 and the BLEU, ImageNet and hotel-limit swaps. Tried and dropped: NLI on focused cells (with or without the caption; the caption makes NLI accept any number), and "field above" form premises (they fixed the form EIN but let a beginning/end swap pass).
- Tests: +4 `test_validator.py`. 228 passing.

### Added — Phase 11 (part 1): Word files ([11-more-formats.md](11-more-formats.md#word-docx))
- `.docx` files are read with Docling's Word reader into one Markdown text: headings become sections, lists are kept, tables become rows (merged headers joined), and equations stay as exact LaTeX. The header and footer go first, once.
- Files without pages (Word, `.txt`, `.md`) are cited by file name and section, no longer "p. 1". `generator.location()`, `pages_label()` and `has_pages()` decide this for the app, both LLM prompts, the checker and the command line.
- LaTeX identical to Docling's "original" (it came from the file, not from a model) is kept without the Phase 9 letter check.
- Old `.doc` files get "save it as .docx or PDF".
- **Eval:** `data/leave-and-travel-policy.docx` (a test file written for this, with a fictional policy) and 9 questions plus un-10. Retrieval: 9/9 in the top 5; reranker R@1 0.82, R@5 0.98 overall. Answers: 9/9 correct, and un-10 declined. One checker false alarm, on a table row with two values.
- Tests: +2 `test_layout.py` (a real Word file; exact LaTeX), +2 `test_loaders.py`, +1 `test_generator.py`. 224 passing.

### Added — Phase 10: OCR for scanned pages, photos and images ([10-ocr.md](10-ocr.md))
- PDF pages without a text layer (fewer than 20 characters) are read with OCR through Docling. Pages with a text layer are never OCR'd: OCR'ing figures on them added a watermark's text and a stray "0" that split a paragraph, and made parsing 5× slower. The four born-digital PDFs are word-for-word unchanged.
- Image files (`.png .jpg .jpeg .tif .tiff .bmp .webp`) load as one-page documents. In the app, the uploader accepts them, and a cited image has a **Show image** toggle.
- **Forms:** a page at least half covered by Docling's "form" region is written row by row from its text cells: `6a Total plan assets   6a   50000   60000`. Phrases join with one space, fields with three, and ticked boxes are written `[x]`. Its header is kept ("Form Number: CA530082"). On 58 pages of papers no form was detected.
- `CITERAG_OCR`: `rapidocr` (default), `ocrmac` (Apple Vision, macOS), `off`. The parse cache key includes the engine, and `PARSER_VERSION` is 2.
- `layout.parse_pdf` → `layout.parse_file` (PDFs and images); `loaders.load_image`. The "no text" warnings say whether OCR ran.
- **Chosen by benchmark** on the user's handwritten form (an IRS 5500-EZ photo). Handwritten values read exactly: RapidOCR 17/17 with rows (14/17 in Docling's order), Apple Vision 15/17, EasyOCR 4/17. On scanned copies of the papers, RapidOCR got 0.966 word similarity with 44 glued words in 16 pages; Apple Vision 0.980 with 4, but it's Mac only.
- **Eval:** `handwriting.pdf` was added with 12 questions and an unanswerable one (91 questions in total). Retrieval: all 12 in the top 5; reranker R@5 0.98. Answers (`gemini-3.5-flash-lite`): 12/12 correct, and both the blank-field question and the off-topic one declined. Off-topic off-08 (an exchange-rate question) now scores 0.452 against the form, just over the 0.45 gate. Kept, because it costs one call and the answer declines.
- Known false alarms: the checker flagged 2 of the 12 correct form answers (a value whose label or column heading is on another row).
- Tests: +5 `test_layout.py` (rows, form pages, the real form, real OCR of an image and a scanned PDF, a mixed PDF where only the scanned page is OCR'd), +3 `test_loaders.py`, `test_ui.py` extended. 219 passing.

### Fixed — while building Phase 10
- Docling numbers `result.pages` from 1; the form text was first sent to a non-existent page 2. Caught by the real-form test.
- Checkbox placement, and word spacing inside handwritten rows.

### System
- No new packages. RapidOCR came with Docling (`rapidocr 3.10.0`, models 31 MB). `onnxruntime` (1.31.0) was already installed through ChromaDB, and is now listed in `requirements.txt` because OCR needs it.
- Benchmark only, in a throwaway environment: `easyocr 1.7.2` and `ocrmac 1.0.1`, plus `onnxruntime` (with `--exclude-newer 2026-11-04`). Not added to the project.

### Added — Phase 9: Layout-aware PDF parsing ([09-layout-parsing.md](09-layout-parsing.md))
- `citerag/layout.py` (new): PDFs go through Docling's layout model (reading order, headings, table structure; OCR off). Each page becomes Markdown with Docling's serializer, with two parts replaced:
  - **Tables:** one `- header: value; …` line per row, caption first. Stacked headers are joined, and merged cells are written once.
  - **Formulas:** the PDF's own text, so a formula is never dropped. With `CITERAG_PDF_FORMULAS=latex`, the formula model's LaTeX is kept only when it has the same letters and digits as the PDF text.
- **Parse cache** in `.cache/parsed/`, keyed by file bytes plus parser version, Docling version and formula mode. A rescan only parses new or changed files.
- `citerag/loaders.py`: `load_pdf(..., parser=)` still checks that the file opens and isn't password-protected with pypdf. pypdf reads the file instead if Docling isn't installed (warned once) or fails on it (warned per file).
- `citerag/config.py`: `PDF_PARSER` (`docling` | `pypdf`), `PDF_FORMULAS` (`text` | `latex`), `PARSE_CACHE_DIR`.
- **Chosen by benchmark** on the 4 PDFs in `data/` with the Phase 8 questions. Docling with rows: hybrid R@5 0.89 → 0.96; reranker R@5 level at 0.97 (R@1 0.84 → 0.79, 3 questions, not significant). Rows beat Markdown tables on every measure.
- **Why not Marker:** 1.10.2 scored reranker R@5 0.84, dropped sentences and wrote numbers as LaTeX. 2.1 needs a system-wide llama.cpp server, and its model weights are restricted for commercial use. Marker was clearly better at formula LaTeX (4/5 vs Docling's 2/5).
- **Answers** (`gemini-3.5-flash-lite`, same settings as Phase 8): 52/55 correct (Phase 8: 53), 0 wrong. The checker's warnings on correct answers fell 7 → 4, because table answers no longer trip it.
- **Speed:** 0.24 s per page after a 2.3 s model load. The formula model takes about 23 s per formula, so it's off by default.
- `eval/questions.json`: 12 quotes that copied pypdf artifacts were rewritten in Docling's form. One figure-label quote was removed (Docling drops text inside figures).
- Tests: `test_layout.py` (10, including a real Docling parse) and 4 new in `test_loaders.py` (the old ones now pin pypdf). 211 passing in total.

### Fixed — while building Phase 9
- The LaTeX check at first rejected the correct two-line PE formula, for two reasons: the `aligned` environment's name counted as letters, and `\\PE` was read as the command `\PE`. Both fixed, and PE is now a test case.
- A benchmark script named `profile.py` shadowed Python's `profile` module (torch imports `cProfile`, which imports `profile`), which deadlocked runs and made Docling look 30× slower than it is. Scratch scripts now run with `python -I`.

### System
- `docling==2.146.0` installed with `uv pip install --exclude-newer 2026-11-03` (packages resolved as of a week earlier; Docling ships almost daily). It adds 48 packages, including `torchvision 0.29.2`, and moves `huggingface-hub` 2.3.0 → 1.33.0. torch, transformers, sentence-transformers and chromadb are unchanged.
- `requirements.txt`: `docling>=2.146,<3`. `.gitignore`: `.cache/`. `.env.example`: the three new settings.
- New local models: Docling layout 328 MB and table structure 683 MB. The formula model (1.2 GB) only downloads in `latex` mode.
- The app's index re-embeds every PDF on the next **Rescan**, because each file's fingerprint is built from its text.

## 2026-11-10

### Fixed — Formulas in answers are rendered ([07-ui.md](07-ui.md#step-7--formulas-2026-11-10))
- Asked for the attention formula, the answer was correct but showed as raw LaTeX (`$$\text{Attention}(Q, K, V) = …$$`), because every `$` was escaped.
- `citerag/ui.py`: `$…$` and `$$…$$` are now passed to Streamlit's KaTeX. A `$$` line becomes a display block; `$$` in running text becomes `\displaystyle`. Other `$` are still escaped (pandoc's rules, so "$5 and $10" stays text), and a `\$` the LLM escaped itself no longer turns back into a math delimiter. `<`/`>` inside a formula become `\lt`/`\gt`. Highlights never cut a formula, and `[n]` inside one isn't made a superscript. Streaming uses the same code.
- Checked in headless Chrome: formulas render, prices stay text, injected HTML and `\href{javascript:…}` do nothing, no console errors.
- Tests: 6 new in `test_ui.py`. 197 passing in total.

## 2026-10-27

### Docs — README rewritten
- `README.md` now explains what CiteRAG does, how the pipeline works (one row per stage, with its code file), requirements (Python 3.12, about 1.4 GB of local models, a Gemini or Anthropic key and the Gemini free-tier limits), setup with uv or pip, using the web app and the command line, the main settings, Phase 8 results, tests, project layout, privacy and limitations.
- Setup now says Python 3.12 and shows the uv route used to build the project, replacing the older "Python 3.10+, venv" instructions.

### Changed — Citation checks tuned on real answers ([08-evaluation.md](08-evaluation.md#step-6--the-citation-checks-on-real-answers))
- `citerag/validator.py`:
  - "Based on the provided documents," lead-ins and a leading "Yes," are removed before NLI (they dropped a correct statement's score from 0.995 to 0.021).
  - The question goes in front of each passage given to the NLI model.
  - A name the question introduces only has to appear in some retrieved passage, so a false premise is still caught.
  - LaTeX becomes plain text (`plain_math`).
  - "10 000" matches "10,000".
  - Bare list items (3 content words or fewer, no digits or negation) are checked by whether the cited passage names them.
  - "do not name/identify" counts as a not-found statement.
- Measured on 170 real statements and 148 deliberately altered ones: true statements warned 48 → 13, altered statements caught 145 → 138, balanced accuracy 0.85 → 0.93. NLI threshold swept 0.3–0.7 and kept at 0.5.
- Tests: 18 new in `test_validator.py`.

### Changed — Reranker on by default ([08-evaluation.md](08-evaluation.md#step-5--answer-benchmark-python--m-evalanswers))
- `USE_RERANKER` 0 → 1 and `RETRIEVE_CANDIDATES` 30 → 10. On the user's documents, the right passage came first 0.56 → 0.84 of the time (sign test p = 0.0005) and answers went 50 → 53/55 correct. Reranking 10 candidates was as accurate as 30 and 3× faster (0.45 s vs 1.4 s per question).
- `app.py` loads the reranker at startup with the other models; UI test checks the preload draws nothing.
- `.env.example`: `CITERAG_USE_RERANKER=0` to turn it off, `CITERAG_RETRIEVE_CANDIDATES`.

### Added — Phase 8: Evaluation ([08-evaluation.md](08-evaluation.md))
- `eval/questions.json`: 78 questions over the syllabus and two papers the user added. 55 answerable (exact evidence quotes, alternatives where a fact appears in several places; facts an answer must contain), 8 unanswerable, 15 off-topic.
- `eval/metrics.py` (quote location, chunk coverage, retrieval and answer scoring), `eval/retrieval.py` (chunk-size × method × candidates sweep, gate calibration, local only), `eval/answers.py` (full pipeline with resumable runs, `--score-only`, `--recheck` with no LLM calls).
- Confirmed: chunk size 384, k = 5, gate 0.45 (0/55 answerable rejected, 0/15 off-topic accepted).
- Found by the first run: gold labels listing only one of several places a fact appears; fixed with alternative quotes and a reachability check.
- Tests: `test_eval.py` (20). 191 passing in total.

### System
- `eval/.cache/` (evaluation indexes) and `eval/runs/` (answers quote the documents) added to `.gitignore`.
- Gemini free tier: `gemini-3.6-flash` allows 20 requests a day (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`). The evaluation runs used `gemini-3.5-flash-lite`, about 124 requests. No new packages; the reranker model was already downloaded in Phase 4.

### Added — Phase 7: Streamlit UI ([07-ui.md](07-ui.md))
- `app.py`: documents sidebar (list, upload into `data/` and index, remove from index, rescan folder); settings (Gemini/Claude, search in, k, check citations, strict mode); streamed answers with superscript citations, unsupported statements highlighted with reasons as tooltips, a summary line, one expander per citation (passage with the quote highlighted, plus the cited PDF page), Checks, and Retrieved passages with scores.
- Plain design, as asked by the user: no emojis or avatars, default theme, minimal toolbar.
- `citerag/ui.py`: safe rendering (LLM/document text HTML-escaped, markdown images removed, `$` escaped), quote highlighting, upload name sanitising, path checks, cutting cited pages out of PDFs.
- `validator.display_offsets`: maps plain-text spans to the displayed answer; strict mode's text cutting now uses it too.
- Tests: `test_ui.py` (17: helpers + headless `AppTest` runs of the real app with fakes); 153 total. Also driven in headless Chrome against the user's syllabus with Gemini.

### Fixed — found by running the app
- Model descriptions were printed on the page (Streamlit "magic" drew the bare preload expressions); regression test added.
- Hundreds of `No module named 'torchvision'` tracebacks per rerun from Streamlit's module watcher: `fileWatcherType = "none"`, plus that logger quietened for Streamlit's reconnect path, which ignores the setting.

### System
- **Security:** the app listened on every network interface (Streamlit's default) and was reachable from the local network. `.streamlit/config.toml` now binds it to `localhost` (verified: `127.0.0.1:8501` only), turns off Streamlit's usage statistics, and uses the minimal toolbar. `.streamlit/secrets.toml` added to `.gitignore`.
- Installed `streamlit-pdf` 2.1.0 via `streamlit[pdf]` (requirements: `streamlit[pdf]>=1.66`); dependency check clean.

### System — Pushed to GitHub
- Committed Phase 6 (citation validation), the marker-parsing fix and the build log to `main`. Before pushing: no Anthropic (`sk-ant-…`) or Google (`AIza…`) keys in changed or new files; `.env`, `.venv/`, `.chroma/` and `data/*` (including the user's syllabus PDF) confirmed git-ignored.

### Added — Phase 6: Citation validation ([06-citation-checks.md](06-citation-checks.md))
- `citerag/validator.py`: each statement of an answer is checked and marked ✅/⚠️ with reasons, plus a summary line. Checks: uncited claims (and whether any retrieved passage supports them), numbers and names missing from the cited passage, support by a local NLI model (cited quotes and passages), invalid `[n]` markers.
- Strict mode (`CITERAG_STRICT=1` / `--strict`) removes the ⚠️ statements and renumbers the remaining citations; it doesn't stream.
- `Answer.validation`, `Answer.invalid_markers` (Gemini's dropped markers are now kept). `Generator(validate=, strict=)`.
- CLI: `Checks:` section after the sources; `--strict`, `--no-validate`; the NLI model loads before the answer streams.
- Config: `VALIDATE` (on), `STRICT` (off), `NLI_MODEL`, `NLI_THRESHOLD=0.5`.
- Tests: `test_validator.py` (27) plus 5 in citations/Gemini; 136 total.

### Decided — NLI model `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, threshold 0.5
- Benchmarked 4 models on RAGTruth QA (795 sentences), 81 hand-labelled syllabus claims and 12 real Gemini answers. With the number/name check: syllabus balanced accuracy 0.953, 3/41 real statements wrongly warned, 0.6 s per answer. tasksource-small was best on RAGTruth (0.752 vs 0.708) but failed on tables; the large MoritzLaurer model was slightly more accurate but about 3× slower.
- Fault injection: 10/10 errors edited into real answers were flagged.

### Fixed — Phase 5 marker parsing ([06-citation-checks.md](06-citation-checks.md#step-5--live-test-and-fault-injection-2026-10-27-gemini-free-tier))
- A `[n]` marker now cites only the last sentence on its own line, not a heading line above it.
- Whitespace before a cited list item is kept in the claim (a first version dropped it); new test: the segments always contain every character of the answer.

### System
- Downloaded 4 NLI models for the benchmark (~2.5 GB in the Hugging Face cache): MoritzLaurer base (used), MoritzLaurer large, tasksource/deberta-small-long-nli, cross-encoder/nli-deberta-v3-base. Downloaded the RAGTruth test split (3.9 MB) to the session scratchpad. No new Python packages.

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
