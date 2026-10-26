# Phase 1 — Document Loaders

**Status:** ✅ Done (2026-10-26) · Code: `citerag/loaders.py` · Tests: `tests/test_loaders.py` (18 passing)

## Goal

Turn PDF, TXT and Markdown files into a list of `Page` records (text + file name + page number), with the text cleaned up enough that the chunker sees normal prose and citations point at the right page.

## The `Page` record

```python
@dataclass
class Page:
    text: str
    source: str   # file name shown in citations
    page: int     # 1-based; TXT/MD files are a single page 1
```

**Why it was created early:** Phase 2 (chunking) was built before Phase 1, and the chunker needed a defined input. Fixing this contract first meant the loader only had to produce a list of `Page` objects.

**Why page-level records:** citations must point to a specific page. If text were loaded as one big string per file, page numbers would be lost before chunking.

## Public API

| Function | Does |
|---|---|
| `load_file(path)` | Picks the right loader from the extension (`.pdf`, `.txt`, `.md`, `.markdown`) |
| `load_pdf(path)` | One `Page` per physical PDF page, using `pypdf` |
| `load_text(path)` | Whole `.txt`/`.md` file as page 1 |
| `load_dir(path)` | Every supported file in a folder (recursive, sorted) |
| `load_paths(paths)` | Any mix of files and folders, used by the CLIs and the later ingestion step |
| `LoaderError` | Raised for unsupported, corrupt or password-protected files |

Preview CLI: `python -m citerag.loaders data/` prints pages and characters per file and the first page of text. The chunker CLI now uses `load_paths` too, so `python -m citerag.chunker report.pdf` works.

## Decisions and why

### 1. Page numbers are physical, as a PDF viewer shows them
A book's printed labels (`iv`, `12`) often differ from the PDF page index. We use the index (1, 2, 3…) because that is what the user types into the viewer's "go to page" box to check a citation. Pages that get skipped keep their real numbers: a document with a blank page 2 yields pages 1 and 3, not 1 and 2.

### 2. Cleanup for every file type (`clean_text`)
| Problem | Fix | Why it matters |
|---|---|---|
| `\r\n`, ` `, form feeds | Become `\n` | Paragraph detection relies on `\n\n` |
| BOM, zero-width characters, control characters | Removed | Invisible characters break exact-match citation checks (Phase 6) |
| Non-breaking spaces | Become normal spaces | Same as above |
| Ligatures `ﬁ ﬂ ﬀ` (very common in PDFs) | Become `fi fl ff` | Otherwise "ﬁnd" never matches a search for "find" |
| Trailing spaces, 3+ blank lines | Trimmed / collapsed to one blank line | Tidy text; blank lines are **kept** because the chunker uses them as paragraph breaks |

Text files are read as UTF-8 (with BOM handled). If that fails, they fall back to Windows-1252 with a warning, since old Windows text files are the most common non-UTF-8 case.

### 3. Extra cleanup for PDFs (`clean_pdf_pages`)
Checked first against real PDFs (Apple license agreements, 4 and 132 pages) to see what `pypdf` actually outputs: one line per printed row, trailing spaces, ligatures, and **no blank lines between paragraphs at all**.

- **Running headers/footers.** The first and last 2 lines of every page are compared across pages, ignoring digits (so "Report — page 3" and "Report — page 4" match). A line is removed if it repeats on at least 3 pages **and** at least half the pages. *Why both thresholds:* in a 2-page document a repeated first line could easily be real content. In the 132-page multilingual agreement, a footer that only closes each language section (a few pages) is correctly kept.
- **Bare page numbers** (`12`, `Page 3 of 10`, `- 4 -`, `xiv`) are removed, but only at the top or bottom of a page. A `42` in the middle of a list stays.
- **Hyphenated line breaks** (`informa-\ntion`) are rejoined when the next line starts in lowercase. Trade-off: a real compound split at a line end (`well-\nknown`) becomes `wellknown`. That is rare and far less harmful than every wrapped word being broken.
- **Paragraph breaks are inferred** (only when pypdf gave no blank lines on that page):
  - a line that ends a sentence (`. ! ? :`) and stops short of the page's text width (under 85% of the 90th-percentile line length) ends a paragraph;
  - a short line (under 60% of the width) without end punctuation, likely a heading or list entry, gets its own paragraph;
  - a line followed by one starting in lowercase is always continued.

  *Why this matters:* the chunker only recognises a heading when it stands as its own paragraph, and prefers to cut chunks at paragraph ends. Without this step a whole PDF page is one paragraph, so section names never reach the chunk headers. On the real agreement, headings like `2. Content and Third-Party Integrations.` now come out as their own paragraphs.

### 4. Failures don't stop a folder load
- Blank or scanned pages (no letters or digits) are skipped with one warning per file listing the pages (`skipped 2 page(s) with no text: 3-4`). A PDF with no text at all gets a "probably scanned (OCR not supported)" warning.
- One broken page inside a PDF is skipped. The rest of the document still loads.
- Corrupt and password-protected PDFs raise `LoaderError`. `load_dir` catches it, warns, and moves on to the next file.
- Hidden files and folders (`.git`, `.DS_Store`, `.chroma`) are ignored.
- pypdf's own logger is set to ERROR, because it warns about every small structural defect it repairs (seen on the GarageBand PDF).

### 5. Sources are relative paths in folder loads
`load_dir("data")` names sources like `reports/q3.pdf`, so two `q3.pdf` files in different subfolders produce different chunk IDs and citations. A single file loaded directly is named by its file name.

## Tests (`tests/test_loaders.py`)

The tests build tiny PDFs in plain PDF syntax (`make_pdf` helper), so they need no extra dependency such as reportlab. They cover:
- one record per page, true page numbers kept after a blank page, the scanned-PDF warning
- header/footer and page-number removal, and headers kept in documents too short to judge
- hyphen rejoining, paragraph and heading inference
- character cleanup, the cp1252 fallback, empty files
- password-protected, corrupt and unsupported files
- folder loading (sorting, relative sources, skipping hidden and broken files), mixed file and folder input
- an end-to-end check: PDF → loader → chunker gives correct `page_start`/`page_end`

**Lesson from a failing test:** the first version of the test PDFs used the same sentence on every page with only the page number changed. The header detector ignores digits, so it treated those body lines as a running header and deleted them. The detector was right; the data was unrealistic. Test pages now use different words.

## Known limitations
- No OCR. Scanned PDFs are reported, not read.
- Multi-column PDFs depend on pypdf's reading order, which can interleave columns.
- Tables come out as plain lines of text.
- A word hyphenated across a **page** break is joined by the chunker with a space (`informa- tion`).
- Very short pages whose lines are nearly identical (forms, templates) can lose lines to header detection.
