# Phase 9 — Layout-aware PDF parsing

**Status:** ✅ Done (2026-11-11) · Code: `citerag/layout.py` (new), `citerag/loaders.py`, `citerag/config.py`; quotes updated in `eval/questions.json` · Tests: `tests/test_layout.py` (10), 4 new in `tests/test_loaders.py`; 211 passing in total

**Outcome in one line:** PDFs are now read with Docling's layout model. Tables become one line per row with every value next to its column name, split words are fixed, and figure labels no longer leak into the text. Hybrid search finds the right passage in the top 5 for 96% of questions instead of 89%; with the reranker it's level (97%). Answers are level (52 vs 53/55 correct, still 0 wrong), and the checker's false alarms on correct answers fell from 7 to 4.

## Goal

The first of three "industry grade" steps (layout-aware parsing, then OCR, then more formats). Phase 8 listed what pypdf gets wrong on the user's documents:

| Problem (pypdf) | Example |
|---|---|
| Table rows come out as runs of numbers, headers far away | `big 6 1024 4096 16 0.3 300K 4.33 26.4 213`: nothing says the 213 is "params ×10⁶" (att-17, missed by every search method) |
| Words split or glued | "a ttend", "dimensiondmodel", "matrixQ", "achieves90.45%" |
| Figure labels mixed into two-column text | `ViT-L 24 layers 4096 4608 5120 …` (axis labels of a ViT figure) |
| Formulas flattened | `Attention(Q, K, V ) = softmax( QKT √dk )V (1)` |

## Step 1 — Candidates

| | Docling 2.146 (IBM) | Marker (Datalab) |
|---|---|---|
| Licence | MIT, code and models | Code Apache-2.0; **model weights modified OpenRAIL-M: free only for research, personal use and companies under $5M** |
| Runs | In process, PyTorch | 2.1 (2026-10-29) needs llama.cpp's `llama-server` installed system-wide (`brew install llama.cpp`) and runs it as a separate process; 1.10.2 is plain PyTorch |
| Also reads | DOCX, PPTX, HTML, XLSX, images, email; OCR built in | DOCX/PPTX/HTML with extras |

Docling also covers the next two steps (OCR, more formats), which Marker's licence and server process would make harder in a company setting. Marker 2.1 couldn't run without installing llama.cpp system-wide, which wasn't done; Marker 1.10.2 was benchmarked instead.

**Supply chain:** every install resolved packages as of a week earlier (`uv pip install --exclude-newer 2026-11-03`), because Docling publishes almost daily (2.150.0 came out the day of the test). Each candidate was installed into its own throwaway environment for the benchmark, so the project `.venv` only got the winner.

## Step 2 — Benchmark on the user's PDFs

Each parser read the 4 PDFs in `data/` (33 pages). The text was scored with the Phase 8 retrieval benchmark (55 answerable questions, the app's chunking and search), plus targeted checks.

**Fair labels first.** The Phase 8 gold quotes were copied from pypdf's text, artifacts included (`big 6 1024 … 213`, `achieves90.45%`). On Docling's text 8 of 62 evidence pieces couldn't be found, so those questions counted as misses before any search ran. Each was given an alternative quote for the **same** evidence in the new format (the same table row, the same sentence), checked by hand. Then:

| Parser | Hybrid R@1 | Hybrid R@5 | Rerank R@1 | Rerank R@5 | Rerank MRR | Evidence findable |
|---|---|---|---|---|---|---|
| pypdf (before) | 0.564 | 0.891 | **0.836** | **0.973** | **0.911** | 62/62 |
| Docling, Markdown tables | 0.573 | 0.891 | 0.764 | 0.945 | 0.856 | 62/62 |
| **Docling, one line per row** | **0.600** | **0.945** | 0.791 | **0.973** | 0.885 | 62/62 |
| Marker 1.10.2 (Markdown tables) | 0.500 | 0.764 | 0.645 | 0.836 | 0.739 | 55/62 |

- **Rows beat Markdown tables** on every measure, so tables are written as rows.
- **Docling with rows vs pypdf:** hybrid search clearly better (top-5 misses 7 → 4). With the reranker, R@5 is the same and three questions dropped from rank 1 to rank 2–3 (vit-04, vit-13, vit-15). The new top passage in each is related but doesn't hold the answer; chunk boundaries moved. 3 losses and 0 gains at rank 1 on 55 questions is not significant (sign test p = 0.25).
- **Marker** lost whole sentences ("We observe that bigger models are more sample efficient" only survives in a figure caption) and wrote numbers as LaTeX (`$10\,000$`), so 7 pieces of evidence couldn't be found even after alternatives.
- **Split and glued words:** pypdf 13 (`matrixQ`, `ratePdrop`, `achieves90.45%` …), Docling 7, all real tokens (`newstest2014`, `bfloat16`). "a ttend" and "dimensiondmodel" are gone.

**Formulas.** Docling's formula model (`do_formula_enrichment`) turns formula images into LaTeX. On the Attention paper:

| Formula | Docling's LaTeX | Marker 1.10's LaTeX |
|---|---|---|
| (1) Attention | `Attent(Q,K,V)=…` (word cut) + stray `& &` | `(Q, K, V) = \operatorname{softmax}…` (word dropped) |
| MultiHead | `MutiHead…` | correct |
| (2) FFN | correct | correct |
| PE (two lines) | correct | correct |
| (3) lrate | `d_{\text{mod}}` for d_model | correct |

Docling also took 113 s for those 5 formulas (about 23 s each, on MPS). So Docling got 2 of 5 right (FFN, PE) and Marker 4 of 5. Marker is clearly better at formulas, and it converts inline symbols (`d_k`) too, but that's its only win. **Decision:** formulas keep the PDF's own text by default, which is faithful. `CITERAG_PDF_FORMULAS=latex` turns the model on, and its LaTeX is kept only when it has exactly the same letters and digits as the PDF text (`same_symbols`). That rejects "Attent", "MutiHead" and "d_mod", and keeps FFN and PE. (The first version of this check also rejected PE, for two reasons. The `\begin{aligned}` wrapper added for KaTeX put the letters of "aligned" into the comparison. And in `\\PE` (a line break, then PE) it read `\PE` as a command and dropped the letters. Environment names are now ignored, control symbols like `\\` are read as one token, and PE is a test case.)

**Speed** (MacBook, MPS, models loaded once in 2.3 s): 33 pages in 8 s, about 0.24 s per page: ViT 2.1 s, Attention 3.7 s, syllabus 1.5 s, WW2 0.7 s. Table structure is most of it. pypdf takes well under a second for all four. Marker 1.10 took 12–174 s per file while sharing the machine.

### A benchmarking mistake worth recording

Early timings said Docling took 78 s and once more than 10 minutes on the Attention paper, with the process idle at 0% CPU. The cause was a scratch script named `profile.py` in the benchmark folder. Python puts a script's own folder first on the import path, torch imports `cProfile`, and `cProfile` imports `profile`: so every script in that folder re-ran the profiling script during `import torch`, deadlocking on the import lock. Renamed, and scratch scripts now run with `python -I` (isolated mode: the script's folder isn't on the path). The real numbers are the ones above.

## Step 3 — Design

**`citerag/layout.py`** turns a PDF into one text per physical page, so the rest of the pipeline (`Page` records, chunker, citations with page numbers) is unchanged.

- Docling options: layout model, table structure in `ACCURATE` mode, OCR off (step 2 of the roadmap), formula model only in `latex` mode.
- Docling's own Markdown serializer writes each page (`MarkdownParams(pages={n})`), with two of its parts replaced:
  - **Tables** (`table_rows`): caption first, then one `- header: value; header: value` line per body row. Header rows are the leading rows with column-header cells (row 0 if none is marked). Stacked headers join per column ("BLEU" over "EN-DE" → "BLEU EN-DE"). Empty cells are left out, and a merged cell is written once. Lines start with `- `, which the chunker already treats as list items, so a long table is split between rows, never inside one.
  - **Formulas:** without the formula model Docling writes `<!-- formula-not-decoded -->` and the formula is lost. Here it keeps the PDF's text (`formula_text`).
- Headings become `## …` (the chunker already reads Markdown headings as sections). Page headers and footers are dropped, because Docling puts them on a separate layer. Text inside pictures isn't written; captions are.
- Checked on the 4 PDFs: no paragraph is written on two pages, and no word is left hyphenated at a line end.

**Parse cache.** The app re-reads the whole folder on every rescan, and a parse takes seconds. Results are stored in `.cache/parsed/<key>.json`. The key is the SHA-256 of the file's bytes plus the parser version, Docling's version and the formula mode, so an edited file, a Docling upgrade or a settings change parses again. A cached file loads in under 0.01 s. Writes go to a temp file and are then renamed, so a crash can't leave half a file. `.cache/` is in `.gitignore`.

**Fallback.** `load_pdf` still opens every PDF with pypdf first: a cheap check that the file opens and isn't password-protected. Then Docling reads it. If Docling isn't installed (warned once) or fails on a file (warned per file), pypdf reads it with the old cleanup. `CITERAG_PDF_PARSER=pypdf` skips Docling entirely.

**Index.** Each file's fingerprint in the vector store is built from its page text, so the first index after this change re-embeds every PDF automatically. Nothing needs resetting.

## Step 4 — Dependencies

`docling==2.146.0` (2026-11-03) added 48 packages, including `torchvision 0.29.2` (matching torch 2.14.1), `docling-parse`, `docling-ibm-models`, `pypdfium2`, `python-docx`, `python-pptx`, `rapidocr` and `mail-parser`, which steps 2 and 3 will use. One package moved: **`huggingface-hub` 2.3.0 → 1.33.0** (Docling's model package requires `<2`). uv resolved it against everything installed, and torch, transformers, sentence-transformers and chromadb are unchanged. The full test suite passes. Models: layout 328 MB, table structure 683 MB (downloaded on first use); the formula model is 1.2 GB and only downloads in `latex` mode.

`requirements.txt`: `docling>=2.146,<3`.

## Step 5 — Evaluation with the production code

`eval/questions.json`: 12 quotes in 11 questions were pypdf artifacts and were rewritten in the new form (for example `big 6 1024 4096 16 0.3 300K 4.33 26.4 213` → `big; N: 6; dmodel: 1024; …; params ×106: 213`). One alternative, `ViT-L 24 layers`, was text inside a figure that Docling now drops, so it was removed (the same question keeps the L/16 table row). The question set now matches the default parser; with `CITERAG_PDF_PARSER=pypdf` those 11 questions would report their quotes as not found.

`python -m eval.retrieval` (production loader, 88 chunks):

| | R@1 | R@5 | MRR | Missed in top 5 |
|---|---|---|---|---|
| Hybrid, Phase 8 → now | 0.56 → **0.62** | 0.89 → **0.96** | 0.73 → **0.78** | 7 → 3 |
| Rerank, Phase 8 → now | 0.84 → 0.79 | 0.97 → 0.97 | 0.91 → 0.88 | att-17, multi-01 (same two) |

The relevance gate is unchanged: at 0.45, 0 answerable questions rejected and 0 of 15 off-topic accepted (off-topic maximum 0.445, as in Phase 8).

`python -m eval.answers --tag docling-lite --no-fallback --delay 7`, with `gemini-3.5-flash-lite` pinned and the reranker on, as in the Phase 8 `rerank-lite` run (62 LLM calls; 16 questions stopped by the gate):

| | Phase 8 (pypdf) | **Phase 9 (Docling)** |
|---|---|---|
| All answerable (55) | 53 correct, 1 partial, 1 refused, 0 wrong | 52 correct, 2 partial, 1 refused, **0 wrong** |
| A cited passage holds the evidence | 54 | 54 |
| The highlighted quote is on the evidence | 48 | 47 |
| Citations to the wrong document | 0 | 0 |
| Unanswerable (8) / off-topic (15): "not in the documents" | 8 / 15 | 8 / 15 |
| **Correct answers the checker warned on** (false alarms) | 7 (syl-07, syl-15, att-14, att-15, vit-08, vit-12, multi-02; after the Phase 8 tuning) | **4** (syl-03, syl-15, vit-08, vit-14) |

- **Answers are level.** The one new partial, att-08, is the model's choice, not a retrieval or parsing loss. The sine-and-cosine sentence was in the top-ranked passage, which Gemini cited, but it only wrote "positional encodings are added to the input embeddings". The Phase 8 answer added the sinusoids. att-17 and multi-01 fail as before.
- **The checker's false alarms fell from 7 to 4.** Three of the old ones were table answers: att-14 (complexity table), att-15 (BLEU table) and vit-12 (ImageNet-v2 in Table 1). With rows, the cited passage says `Layer Type: Self-Attention; Complexity per Layer: O(n2 · d)` instead of a run of numbers, and NLI accepts the statement. Phase 8 found tables were the largest group of false alarms.
- Single runs of a non-deterministic model: differences of one question are noise. The false-alarm drop is consistent with where the old alarms were.

## Step 6 — Tests

**`tests/test_layout.py` (10)**, on Docling documents built in code (fast, offline):
- table rows name their columns; blank cells are left out
- stacked headers join, and merged cells are written once
- a table without header flags uses its first row
- LaTeX is kept only when it matches the PDF text, using the formula model's real output for the Attention paper (FFN and the two-line PE kept; "Attent" and "d_mod" rejected)
- LaTeX cleanup: compact spacing, the equation number removed, a space kept after a command, words inside `\text{}`, an aligned environment for multi-line formulas
- `same_symbols`
- one Markdown text per page
- the parse cache: by content and by settings
- conversion errors become `LayoutError`; an unknown formula mode is refused
- **real Docling** on a generated 3-page PDF, with a blank middle page and no text leaking between pages (skipped if Docling isn't installed)

**`tests/test_loaders.py` (+4):**
- Docling pages are cleaned and numbered, and blank pages skipped
- pypdf takes over when Docling fails (warned per file) or isn't installed (warned once)
- the password check runs before Docling
- an unknown parser is refused

The existing loader tests now pin `pypdf`: they test its cleanup.

## Known limitations

- **att-17 is still missed.** The row now reads `big; …; params ×106: 213`, but the question says "parameters" and the table says "params", and nothing in the row says "Transformer". Retrieval would need query expansion or a table-aware index.
- **Superscripts are flattened** the same way as before: `×106` is ×10⁶, `1019` is 10¹⁹, `2242` is 224².
- **The first column of a table often has no header** (Table 3's `base`/`big`), so its value appears without a label.
- **Scanned PDFs** still have no text: OCR is off until the next step. *(Done in Phase 10: [10-ocr.md](10-ocr.md).)*
- **Formulas** stay as PDF text unless `latex` mode is on, and then only matching LaTeX is kept.
