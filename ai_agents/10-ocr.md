# Phase 10 — OCR for scanned PDFs, photos and images

**Status:** ✅ Done (2026-11-11) · Code: `citerag/layout.py`, `citerag/loaders.py`, `citerag/config.py`, `citerag/ui.py`, `app.py`; form questions added to `eval/questions.json` · Tests: 5 new in `tests/test_layout.py`, 3 in `tests/test_loaders.py`, 1 extended in `tests/test_ui.py`; 219 passing in total

**Outcome in one line:** scanned PDF pages, photos of documents and image files are now read with RapidOCR. Forms are read row by row, so each value stays next to its label. On the user's handwritten test form, all 17 checkable handwritten values are read exactly, and all 12 questions about it are answered correctly with citations.

## Goal

Step 2 of the three "industry grade" steps (layout parsing, **OCR**, more formats). Before this phase, a PDF without a text layer was skipped with "no text found", and images couldn't be added at all.

## Step 1 — The test document

The user added `data/handwriting.pdf`: one page, no text layer. It's a **photo** (not a flatbed scan) of an IRS Form 5500-EZ filled in by hand with sample data ("Acme Corp Software", made-up ID numbers). It mixes three hard things:
- printed form text, photographed with shadows and a slight tilt
- handwritten values: dates, names, nine-digit numbers, dollar amounts, a ticked checkbox
- a form layout, where a value means nothing without its label

**Ground truth** was transcribed by reading the page: 21 handwritten values (17 of them 3+ characters long, so they can be checked by search; "2" or "5" occur anywhere on the page) and 12 printed reference lines. Scores: a value is *exact* if it appears in the text (ignoring case, spaces and punctuation); *similarity* is the best match of the value against any stretch of the text.

## Step 2 — Engines

Docling can drive several OCR engines. On this Mac:

| Engine | Notes |
|---|---|
| RapidOCR (PP-OCRv6 models on ONNX) | Any OS; Apache-2.0; already installed with Docling; models 31 MB |
| EasyOCR | Any OS; PyTorch |
| Apple Vision (`ocrmac`) | Built into macOS; Mac only, so not for a Linux server |

Not tested: **Nemotron OCR** (needs Linux with an NVIDIA GPU) and **Tesseract** (a system install, and known to be weak on handwriting).

**Handwritten form, Docling's normal reading order:**

| Engine | Handwritten values exact (of 17) | Similarity, handwritten | Similarity, printed | Time |
|---|---|---|---|---|
| RapidOCR | 14 | 0.964 | 0.961 | 3.8 s |
| Apple Vision | 13 | 0.965 | 0.961 | 1.7 s |
| EasyOCR | 4 | 0.831 | 0.680 | 6.7 s |

EasyOCR is out. But the bigger problem wasn't recognising characters.

## Step 3 — Forms lose their labels

In Docling's reading order, the form fell apart:
- **Line items 5–6 became a loose column of numbers**, printed after all the labels: `5732900 10 … 8 5 2 … 50000 4000 60000 $5000`. Nothing says which number is "total plan assets, end of year".
- The EIN `735268329` landed at the end of the page, far from "2b Employer Identification Number".
- **"Form Number: CA530082" was dropped**: Docling classified the handwritten top line as a running page header. That's right for papers and wrong here.

On a form, a label and its value share a **visual row** ("6a Total plan assets …… 6a $50000 $60000"). So rebuilding a page row by row from the text boxes was tested:

| Form text | Values exact (of 17) | Printed similarity |
|---|---|---|
| Docling order, RapidOCR | 14 | 0.961 |
| RapidOCR boxes on the page image, by rows | 16 | 0.988 |
| **Docling's text cells, by rows, RapidOCR** | **17** | 0.986 |
| Docling's text cells, by rows, Apple Vision | 15 | 0.988 |

**But row order destroys multi-column pages** (Step 4), so it can't be used everywhere. Docling's layout model has a dedicated **"form" region**. On the handwritten page it covers 76% of the page; on all 58 pages of the four papers and their scanned copies it covers 0%. **Rule: a page at least half covered by a form region is written row by row; every other page keeps Docling's order.**

Row details (`rows_text`): a box joins a row when its vertical centre is within half a box height of the row's. Inside a row, boxes closer than the row's height are words of one phrase (one space), and wider gaps separate fields (three spaces). The quote locator matches by words and line, so the spacing doesn't affect citations. Ticked and unticked boxes, which Docling detects as their own regions, are written `[x]` and `[ ]` in front of their item: `A This return is: [x] (1)the first return filed for the plan   [ ] (3)the final return filed for the plan`. On form pages the page header and footer are kept.

## Step 4 — Scanned papers

The three evaluation PDFs were turned into scans: every page rendered at 200 DPI, grey, tilted ±0.3–0.4°, JPEG quality 70, saved as image-only PDFs. Their correct text is the Phase 9 text, so the Phase 8 questions apply unchanged.

| Text | Word similarity to the original | Evidence quotes lost | Reranker R@5 |
|---|---|---|---|
| Born-digital (text layer) | 1.00 | 0 | 0.973 |
| **Scan → Docling order + RapidOCR** | **0.96–0.97** | 9 | 0.80 |
| Scan → page-wide rows | 0.47–0.93 (two-column ViT paper: 0.47) | 17 | 0.66 |

- **Page-wide rows interleave the two columns**, hence the form-only rule above.
- **Almost all the lost quotes are dropped spaces:** `marks:100`, `courseno`, `universalselector`, `nearly3`, `inetv2`. The words are right; the evidence check needs an exact match, so part of the R@5 drop is strict scoring rather than retrieval. Dropped spaces do hurt keyword search, though.

**Settings, on the scanned syllabus and Attention paper (16 pages):**

| Engine | Word similarity | Glued words | Speed |
|---|---|---|---|
| RapidOCR small (Docling's default) | 0.966 | 44 | 2.1 s/page |
| RapidOCR small, English | 0.966 | 44 | 2.2 s/page (same model) |
| RapidOCR medium | 0.980 | 20 | 8–10 s/page |
| Apple Vision | 0.980 | **4** | **0.8 s/page** |

**Decision:** RapidOCR small is the default. It's best on the handwritten form (17/17), it runs anywhere (a server will be Linux), and it adds no new packages. Apple Vision is better on printed scans and faster, so it's a setting for Macs: `CITERAG_OCR=ocrmac` (`pip install ocrmac`). The medium model halves the glued words at 4× the time, which isn't worth it for uploads in the app.

## Step 5 — OCR only where there's no text layer

The first version let Docling OCR every image area, including figures on born-digital pages. Compared with the Phase 9 text:
- The syllabus took 7.4 s instead of 1.5 s.
- The ViT paper gained a watermark image's text ("This CVPR paper is the Open Access version…") and a stray `0` from a figure. The `0` stopped Docling joining a paragraph that runs across pages 2–3, so it was split in two.

A text layer is already exact, so **only pages without one are OCR'd**. `text_layer_chars` counts each page's text-layer characters with `pypdfium2`, which Docling already depends on. A page with fewer than 20 characters is treated as a scan. The PDF is converted once without OCR, then once with OCR over the span of scanned pages, and only those pages are taken from the second pass. Images are always OCR'd. After the change, all 33 pages of the four born-digital PDFs are identical to their Phase 9 text.

## Step 6 — Wiring

- `config.OCR` (`CITERAG_OCR`): `rapidocr` (default), `ocrmac`, or `off`. An unknown value is refused, and `ocrmac` without the package says how to install it.
- `layout.parse_file` (was `parse_pdf`) takes PDFs and images. The parse cache key now includes the OCR engine, and `PARSER_VERSION` is 2, so cached Phase 9 parses are redone.
- `loaders`: `.png .jpg .jpeg .tif .tiff .bmp .webp` are supported (`load_image`: one page). An image without Docling or with OCR off gives a clear error. The "no text" warning now says whether OCR ran.
- App: the uploader accepts images. A cited image has a **Show image** toggle, like **Show PDF**. The "no text found" message no longer says OCR isn't supported.
- `requirements.txt`: `onnxruntime` is listed explicitly. RapidOCR needs it, and until now it only came in because ChromaDB needs it too.

## Step 7 — Evaluation

**Eval set:** `handwriting.pdf` is now an evaluation document, with **12 questions** (form-01 to form-12): EIN, plan number, effective date, employer, phone, participants, assets and liabilities, form number, the ticked checkbox, administrator's EIN, plan year. Plus **un-09**, the administrator's telephone number, which was left blank on the form. It forbids the employer's number in the answer, the most likely mix-up. form-01 and form-11 also forbid each other's EINs. 91 questions in total, 67 answerable.

**Retrieval** (`python -m eval.retrieval`, 4 documents, 91 chunks): all 12 form questions have their evidence in the top 5. Reranker R@1 0.80, R@5 0.98; still missing att-17 and multi-01, as before.

**The gate:** one off-topic question now passes 0.45. off-08, "What is the exchange rate between the US dollar and the euro?", scores 0.452 (0.441 before) against the form's dollar amounts. The threshold stays: moving it for one question would be fitting to noise. The cost is one LLM call, and the answer was "not in the documents" (below).

**Answers** (`gemini-3.5-flash-lite`, as in Phases 8–9; `--only` the 14 new or affected questions):

| | Result |
|---|---|
| Form questions (12) | **12 correct**, 0 partial, 0 wrong; a cited passage holds the evidence in 12, the highlighted quote is on it in 10 |
| un-09 (blank administrator phone) | declined, didn't give the employer's number |
| off-08 (exchange rate, passed the gate) | declined |
| Checker warnings on correct answers | 2 of 12 (form-01, form-08) |

The two warnings are **false alarms caused by the form layout**. The EIN sits two rows below its label, beside "Trade name of business". And the "Beginning of year" column heading is two rows above `6b 4000 $5000`. The checker judges each claim against the cited passage and reads it as a contradiction. It's the same kind of problem tables had before Phase 9, where values were cut off from their headers.

## Problems found while building

1. **Docling numbers `result.pages` from 1**, not 0 as assumed, so the form text went to a page 2 that doesn't exist. The real-form test caught it on the first run.
2. **Word-level cells** in the handwritten header came out as `Form   Number:   CA530082`. Fixed with the gap rule (phrase vs field).
3. **`[x]` after its item:** the OCR box for "(1)the first return…" starts left of the drawn box. A checkbox inside a text box's span now goes first.
4. **OCR on every image area** (Step 5).
5. A stale parse cache from development runs, written under the same version, briefly hid the Step 5 fix. Cleared locally; nothing was released with it.

## Tests

**`tests/test_layout.py` (+5):**
- `rows_text`: rows read left to right, a phrase vs separate fields, taller handwriting boxes, empty boxes dropped
- `form_pages`: only pages at least half covered by a form; the header kept; checkboxes in place, also when drawn inside a line's box
- **real form**: `data/handwriting.pdf` read with RapidOCR keeps "Form Number: CA530082" on line 1, line 6a with both amounts, 5a(1) with 10, and the IDs (skipped if the file isn't there)
- **real OCR**: a generated PNG and the same image as an image-only PDF are both read; with OCR off the scan has no text
- **real mixed PDF**: two text pages and one scanned page; only the scan is OCR'd, and the text pages are identical to an OCR-off parse

Also updated: the cache test (the OCR engine is part of the key), and the settings errors (unknown engine; `ocrmac` not installed).

**`tests/test_loaders.py` (+3):** images are page 1 and found in folders; images need Docling and OCR; the "no text" warning says whether OCR ran.

**`tests/test_ui.py`:** `source_image` stays inside the folder, and PDFs and images aren't confused.

## Known limitations

- **One handwritten test page.** 17/17 on it is encouraging, not proof. Messier handwriting, or other scripts (Nepali/Devanagari wasn't tested), may do much worse.
- **RapidOCR drops spaces** on printed scans (44 glued words on 16 pages), which hurts keyword search. Apple Vision (`CITERAG_OCR=ocrmac`) glues only 4, on a Mac.
- **Scanned papers retrieve worse than born-digital ones** in the benchmark (R@5 0.80 vs 0.97), partly because of glued words and partly because of exact quote matching.
- **The checker and forms:** values whose label or column heading is on another row can be flagged as contradicted (2 of 12).
- **Tiny print under handwriting** comes out garbled ("Ciyte r o dor e e o" for "City or town, state…"). It's printed instructions, not data.
- **Images inside text pages aren't OCR'd** (a scanned letter pasted into a Word export, for example). That's deliberate (Step 5).
- **Form detection is Docling's.** A form it doesn't recognise keeps its normal reading order.
