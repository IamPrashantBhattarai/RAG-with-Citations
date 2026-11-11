# Phase 11 — More file formats

**Status:** 🔄 In progress: Word (.docx) done (2026-11-11). Next: PowerPoint, HTML and web pages, email · Code so far: `citerag/layout.py`, `citerag/loaders.py`, `citerag/generator.py`, `citerag/gemini.py`, `citerag/validator.py`, `app.py` · Tests so far: +2 `tests/test_layout.py`, +2 `tests/test_loaders.py`, +1 `tests/test_generator.py`; 224 passing in total

## Goal

Step 3 of the three "industry grade" steps: read the document types people actually keep, one file type at a time. Each gets a realistic test file, tests, and questions in the eval set.

## Word (.docx)

### What Docling gives

Docling reads `.docx` directly from the file's structure. No layout model or OCR is needed, because a Word file already says what is a heading, a list or a table. The test file `data/leave-and-travel-policy.docx` was written for this phase (a fictional company's leave and travel policy, with amounts in NPR) and has the parts that trouble parsers:
- a title and headings at two levels
- a numbered list and a bulleted list
- a table whose header is merged two ways (a column header spanning two rows, and "Hotel limit per night (NPR)" spanning the "Kathmandu" and "Other cities" columns)
- a running header ("Northwind Analytics · Internal · Version 3.2") and footer ("Confidential — document number HR-POL-014")

Docling with CiteRAG's Phase 9 serializers gets all of it right:
- headings become `#`, `##` and `###`, so the chunker sets sections
- lists are kept
- the merged table header is joined correctly: `- Grade: A (Director); Daily allowance (NPR): 6,000; Hotel limit per night (NPR) Kathmandu: 18,000; Hotel limit per night (NPR) Other cities: 12,000`

### Decisions

**No pages: cite the file and section, not "p. 1".** A Word file has no fixed pages; they depend on the printer, fonts and paper size. The whole file is one text, stored as page 1 so the rest of the pipeline is unchanged. Citations and prompts then show only the file name, with the section beside it, instead of an invented "p. 1". The same now applies to `.txt` and `.md`, whose "p. 1" never meant anything either. `generator.location()` / `pages_label()` / `has_pages()` are the one place this is decided; the app, the Gemini prompt, Claude's document titles, the checker's messages and the command line all use them.

**Header and footer: kept once, at the top.** For PDFs, Docling drops running headers and footers, which is right: they repeat on every page. A Word file stores them once, and here the footer holds the document number, something people ask about. They go before the first heading, so no section claims them. (Docling's own Markdown puts them at the end, inside the last section.)

**Equations: Word's LaTeX is kept.** Word stores equations as structured math (OMML), which Docling turns into exact LaTeX. Both its `text` and `orig` hold that LaTeX. The Phase 9 check keeps formula-model LaTeX only when it matches the PDF's own text, and it would have rejected `\frac{a}{b}` (the letters "frac" aren't in any PDF text). So when `text` and `orig` are identical, the LaTeX came from the file and is kept as it is.

**Old `.doc` files** (Word 97-2003) can't be read by Docling. They get a clear message ("save it as .docx or PDF") instead of "unsupported file type".

### Evaluation

The eval set gains `leave-and-travel-policy.docx` with 9 questions (docx-01 to docx-09): days of leave, carry-over, notice period, who approves and how fast, two table cells (Grade B's daily allowance, a Director's hotel limit in Kathmandu), the sick-note rule, the approver for expensive trips, and the document number in the footer. There is also **un-10** (paternity leave, which the policy doesn't cover). 101 questions in total, 76 answerable.

- **Retrieval** (5 documents, 93 chunks): all 9 in the top 5. Reranker R@1 0.82, R@5 0.98; the misses are still att-17 and multi-01. The gate is unchanged.
- **Answers** (`gemini-3.5-flash-lite`, `--only` the 10 new questions): **9/9 correct**. The cited passage holds the evidence in 9, and the highlighted quote is on it in 7. un-10 was declined. Every citation reads `leave-and-travel-policy.docx`, with no page.
- **Checker:** 1 false alarm (docx-06, the Kathmandu hotel limit). The row has two hotel limits, and NLI reads the claim as contradicted: the same table weakness as before.

### Tests

- `test_layout.py`: a real Word file built with python-docx is read as one text, with the header and footer first, sections, the numbered list, the merged table header joined, and an OMML equation kept as `$$a^{2}+b^{2}$$`. Also: LaTeX identical to its original is kept, while model LaTeX that disagrees isn't.
- `test_loaders.py`: `.docx` loads as one page-1 text and needs Docling; `.doc` says to save as .docx or PDF.
- `test_generator.py`: PDFs and images keep "p. 3"; Word, Markdown and text files are cited by name only, including in Claude's document titles.

### Limitations

- **Pictures inside a Word file aren't read.** A scanned page pasted into a .docx is skipped (Docling's Word reader doesn't OCR them).
- **Comments, tracked changes and text boxes** are whatever Docling's Word reader gives; they weren't tested.
- **Long Word files have no page numbers in citations.** The section heading is the only pointer.
