# Phase 11 — More file formats

**Status:** 🔄 In progress: Word (.docx), PowerPoint (.pptx), HTML files and web pages done (2026-11-11). Next: email · Code so far: `citerag/layout.py`, `citerag/loaders.py`, `citerag/chunker.py`, `citerag/store.py`, `citerag/generator.py`, `citerag/gemini.py`, `citerag/validator.py`, `app.py` · Tests so far: +5 `tests/test_layout.py`, +3 `tests/test_loaders.py`, +1 `tests/test_generator.py`, +1 `tests/test_chunker.py`, `tests/test_web.py` (4, new); 237 passing in total

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

## PowerPoint (.pptx)

### What Docling gives

Docling reads `.pptx` from the file's structure, one page per slide. The slide title becomes a `#` heading, bullets are kept, and tables become one line per row. **Speaker notes** are found but put in a separate "notes" layer that the default output leaves out. The test deck `data/q3-business-review.pptx` was written for this phase (a fictional company's quarterly review). It has a title slide, bullet slides with numbers, a revenue-by-region table, and speaker notes holding a fact that isn't on any slide ("The board approved hiring two data engineers in October").

### Decisions

- **Speaker notes are kept,** after their slide, marked "Speaker notes:". Presenters put there what they say but don't show.
- **Citations say "slide 3"**, not "p. 3" (`generator.pages_label`).
- **One chunk per slide.** The first run packed slides 1–4 into one chunk. Citations said "slides 1–4", every chunk took the deck title as its section, and the speaker-notes question fell below the relevance gate (best cosine 0.431 < 0.45): one sentence of notes was diluted among four slides. Worse, `chunker._join_pages` joins pages with a space when a page doesn't end with a full stop, which suits PDF paragraphs running across pages but not slides, which rarely end with one. So "# Q3 highlights" landed mid-line and wasn't read as a heading. Now slides always start a new paragraph, and a chunk never spans two slides; a long slide is still split. The notes question's cosine went from 0.431 to 0.527.
- **`CHUNKER_VERSION`** (new, now 2) is part of each file's index fingerprint. Before, the fingerprint only covered the file's text and the chunk settings, so a change in chunking *logic* left old chunks in existing indexes. Bumping it re-chunks and re-embeds every file once.
- **Old `.ppt` files** get "save it as .pptx or PDF".

### Evaluation

9 questions (pptx-01 to pptx-09): revenue, growth, a new client, churn, two table questions (the fastest-growing region, and Kathmandu's Q3 revenue, competing with the Word file's Kathmandu hotel limits), the Q4 target, the launch month, and the speaker-notes fact. Plus **un-11** (the CEO's name). 111 questions in total, 85 answerable.

- **Retrieval** (6 documents, 98 chunks): all 9 in the top 5. Reranker R@1 0.83, R@5 0.98; the misses are still att-17 and multi-01. The other documents' chunks are unchanged (same counts). Gate: 0 answerable questions rejected. off-08 (an exchange-rate question) now scores 0.556 against the "Risks" slide, which really mentions exchange-rate exposure, so it passes the gate.
- **Answers** (`gemini-3.5-flash-lite`): **9/9 correct**. Each cites the exact slide, and the highlighted quote is on the evidence in all 9. **No checker warnings:** the two table answers were confirmed by the new table rule. un-11 and off-08 were both declined.

### Tests

- `test_layout.py`: a real deck built with python-pptx gives one text per slide: the title as a heading, bullets, a table row, the slide's speaker notes, and "" for a blank slide.
- `test_loaders.py`: one page per slide (blank slides skipped, numbering kept); `.ppt` says to save as .pptx or PDF.
- `test_chunker.py`: a deck gets one chunk per slide with the slide titles as sections, while the same pages as a PDF are packed together; slides always join with a paragraph break.
- `test_generator.py`: "slide 3", "slides 3-4".

### Limitations

- **Charts, SmartArt and images on slides aren't read** (a screenshot of a table, for example). OCR on embedded images is the open item noted under Word.
- **Slide order is file order;** hidden slides are read like the others.

## HTML files and web pages

### HTML files (.html, .htm)

Read with Docling's HTML reader as one text, cited by file name and section, like Word. Two differences from Word:
- **No header/footer block.** In HTML, Docling's "furniture" is the page title, menus, cookie banner and footer, which is clutter, not a document number.
- **Link targets are dropped** (`[Shop](/shop)` → `Shop`), for Word too. Inline links are joined with a space by Docling ("See prices ."); a space between a word and punctuation is removed in Word and HTML text.

**Main-content cleanup (`layout.clean_html`).** On a tidy test page, Docling's own furniture detection was enough. A real Wikipedia article (Net metering) still brought its table of contents, a 17-language menu, "Edit links", "Toggle" buttons and `[ 1 ]` / `[ note 1 ]` reference markers. The markers are a real risk, because the LLM cites with `[n]` markers and could copy them. Before Docling, a page is now reduced with BeautifulSoup (already installed with Docling):
- keep `<main>` (or `role="main"`, or `<article>`) when the page has one
- remove clutter tags (`nav`, `aside`, `form`, `button`, `script`, `svg`, `iframe`…) and hidden elements (`hidden`, `aria-hidden="true"`)
- remove elements whose id or class has a clutter **word** (`toc`, `menu`, `cookie`, `lang`, `toolbar`, `editsection`, `navbox`, `catlinks`, `share`, `reference`…), matched as whole words so `stock-price` survives `toc`

On the Wikipedia page that took 56,566 characters of mostly clutter down to 29,301, starting with the article's title and first paragraph, with 0 reference markers left.

### Web pages (`citerag/web.py`, new)

A URL is downloaded once and **saved into `data/`** as a file (`example.org-blog-solar-guide.html`), so it's indexed, cited and rescanned like any document, and answers keep working if the page changes or disappears. Saved HTML starts with `<meta name="citerag-source-url" …>` and `<meta name="citerag-saved" …>`, and the loader puts "Web page: https://…" at the top of its text. A URL that serves a PDF is saved as a `.pdf`.

**Safety.** The app runs on the user's computer, so a URL must not make it read the local network (SSRF):
- only `http`/`https`
- **every address the host resolves to must be public** (`ipaddress.is_global`: no localhost, 10.x, 192.168.x, 100.64.x, `fd00::/8`, or the `169.254.169.254` cloud metadata address)
- checked again at **every redirect** (redirects are followed by hand, at most 5)
- 15 s timeout, a 5 MB cap (checked on the header and while reading), and only `text/html`, `application/xhtml+xml` or `application/pdf`
- written to a temp file and renamed, so a failed download leaves nothing
- a generic User-Agent (`CiteRAG/1.0`), no personal details

**App:** "Add a web page" in the sidebar (URL → *Save and index*). **CLI:** `python -m citerag.web URL`.

Checked live: the Wikipedia page saved and parsed in 2.2 s; `http://localhost:8501` and `http://169.254.169.254/latest/meta-data/` were refused with "… is not a public internet address".

### Evaluation

`data/rooftop-solar-guide.html` is a test page written for this phase (a fictional solar blog post with a header, menu, cookie banner, sidebar, footer and scripts). It gets 6 questions (html-01 to html-06: the installed price, payback, household use, the buy-back rate, a table value, the meter type) and **un-12** (panel warranty, not covered). 118 questions in total, 91 answerable.

- **Retrieval** (7 documents, 100 chunks): all 6 in the top 5. Reranker R@1 0.84, R@5 0.98; still missing att-17 and multi-01. The gate is unchanged.
- **Answers** (`gemini-3.5-flash-lite`): **6/6 correct**, no checker warnings; un-12 was declined. The free tier was throttling at this point: 7 calls took about 20 minutes, waiting and retrying (`--no-fallback`), and no answer was lost.

### Tests

- `test_web.py` (4, new): a local HTTP server stands in for the web.
  - A page is saved with its address and read back as "Web page: …" without its menu; a same-site redirect is followed; a PDF URL is saved as `.pdf`.
  - Refusals say why: a redirect to `localhost`, an image, a 404, `ftp://`, loopback without the test's allowance, and an oversized page, with nothing half-saved.
  - Private, loopback, link-local, ULA and carrier-grade NAT addresses are rejected, and one private address among public ones is enough to refuse.
  - File names come from the address.
- `test_layout.py`: real Docling HTML keeps the article without the menu, cookie banner, footer or link targets; `clean_html` keeps `<main>`, drops clutter by whole word, keeps `stock-price`, drops hidden elements and footnote markers.

### Limitations

- **The cleanup is a heuristic.** A site that puts content in a div called "menu", or clutter in unmarked divs, will lose or keep the wrong parts.
- **DNS can change between the check and the download** (DNS rebinding); the window is small, and a later redirect is checked again.
- **Pages built by JavaScript** (single-page apps) have little HTML to read, because no browser runs here.
- **Logins and paywalls** aren't handled; you get what an anonymous visitor gets.

