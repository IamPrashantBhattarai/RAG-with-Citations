"""Layout-aware PDF parsing with Docling. (Phase 9)

pypdf returns a PDF's text in drawing order, so tables come out as runs of
numbers ("big 6 1024 4096 16 0.3 300K 4.33 26.4 213"), symbols run into
words ("dimensiondmodel"), and figure labels mix into two-column text.
Docling runs a layout model over each page instead. It finds the reading
order, headings, paragraphs, tables (with their cell structure), formulas,
figures and page headers/footers. That becomes one text per page:

- Headings become markdown "## ", so the chunker sees sections.
- Every table row becomes one line that names its columns, e.g.
  "- big; N: 6; dmodel: 1024; ...; params ×106: 213". A row then makes sense
  on its own: in search, in a chunk cut between rows, and to the citation
  checker. The table's caption goes first.
- Formulas keep the PDF's own text. With CITERAG_PDF_FORMULAS=latex, Docling's
  formula model writes LaTeX instead, kept only when it has the same letters
  and digits as the PDF text (on the Attention paper it misread 3 of 5:
  "Attent", "MutiHead").
- Page headers/footers and the text inside figures are dropped. Captions stay.

OCR (Phase 10): PDF pages without a text layer (scans, photos) are read with
RapidOCR (CITERAG_OCR; Apple Vision with "ocrmac" on a Mac, or "off"). Pages
with a text layer keep it and are not OCR'd. Image files (PNG, JPEG, TIFF, …)
are read with OCR as one-page documents.

Forms (Phase 10): on a page that is mostly a form (Docling's "form" region
covers at least half of it), Docling's reading order separates values from
their labels, e.g. a column of amounts printed after all the line items. Such
a page is written row by row instead, reading each visual row left to right:
"6a Total plan assets   6a   50000   60000". The header is kept on these
pages: a form's top line is content ("Form Number: CA530082").

Parsing takes seconds per page plus a model load, so results are cached on
disk by file content and parser settings. Re-scanning a folder only parses
new or changed files.
"""

from __future__ import annotations

import hashlib
import html
import importlib.metadata
import importlib.util
import json
import logging
import os
import re
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from citerag import config

log = logging.getLogger(__name__)

# Bump when the text this module produces changes, so cached parses are redone.
PARSER_VERSION = 2
FORMULA_MODES = ("text", "latex")
OCR_ENGINES = ("rapidocr", "ocrmac", "off")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
# Formats without fixed pages (Phase 11): read as one text, cited by section rather than page.
UNPAGED_EXTENSIONS = {".docx", ".html", ".htm"}
HTML_EXTENSIONS = {".html", ".htm"}
# Saved web pages (citerag/web.py) carry the address they came from in this meta tag.
_SOURCE_URL_RE = re.compile(rb'<meta name="citerag-source-url" content="([^"]+)"')
_SPACE_BEFORE_PUNCT_RE = re.compile(r"(?<=\w) (?=[.,;:!?](?:\s|$))")
# Formats whose "pages" are slides (Phase 11): one text per slide, cited as "slide 3".
SLIDE_EXTENSIONS = {".pptx"}
FORM_PAGE_SHARE = 0.5  # a page this much covered by a "form" region is written row by row
MIN_TEXT_LAYER_CHARS = 20  # a PDF page with fewer text-layer characters than this is OCR'd

_lock = threading.Lock()  # one conversion at a time: the models aren't shared safely
# Function-like commands that print as words, so they count when comparing with the PDF text.
_LATEX_WORDS = {"max", "min", "sin", "cos", "tan", "log", "ln", "exp", "lim", "sup", "inf", "det", "arg"}
_LATEX_TEXT_RE = re.compile(r"\\(?:text|mathrm|operatorname|mathit|mathbf|textbf|mathsf|textit)\s*\{([^{}]*)\}")
_LATEX_TOKEN_RE = re.compile(
    r"\\(?:text|mathrm|operatorname|mathit|mathbf|textbf|mathsf|textit)\s*\{[^{}]*\}"
    r"|\\[A-Za-z]+|\\.|\S"
)
_EQ_NUMBER_RE = re.compile(r"(?:\s*&)*\s*\(\s*\d+\s*\)\s*$")  # "… & & ( 1 )" at the end


class LayoutError(Exception):
    """Docling could not convert the file."""


def available() -> bool:
    return importlib.util.find_spec("docling") is not None


def parse_file(path: str | Path, formulas: str | None = None, ocr: str | None = None) -> list[str]:
    """The text of each page of a PDF or image (index 0 = page 1), from cache when possible."""
    formulas = formulas or config.PDF_FORMULAS
    ocr = ocr or config.OCR
    if formulas not in FORMULA_MODES:
        raise ValueError(f"CITERAG_PDF_FORMULAS must be one of {FORMULA_MODES}, not {formulas!r}")
    if ocr not in OCR_ENGINES:
        raise ValueError(f"CITERAG_OCR must be one of {OCR_ENGINES}, not {ocr!r}")
    if ocr == "ocrmac" and importlib.util.find_spec("ocrmac") is None:
        raise ValueError("CITERAG_OCR=ocrmac needs the ocrmac package (macOS only): pip install ocrmac")
    path = Path(path)
    data = path.read_bytes()
    cache = _cache_path(data, formulas, ocr)
    if cache.is_file():
        try:
            return json.loads(cache.read_text())["pages"]
        except (OSError, ValueError, KeyError):
            log.warning("%s: unreadable parse cache, parsing again", path.name)
    with _lock:
        try:
            pages = _convert(path, formulas, ocr)
        except Exception as e:  # Docling raises many types; the loader falls back to pypdf
            raise LayoutError(str(e) or type(e).__name__) from e
    _write_cache(cache, pages)
    return pages


def _convert(path: Path, formulas: str, ocr: str) -> list[str]:
    """Images are read with OCR. PDF pages with a text layer keep it; only pages without one are OCR'd.

    OCR on a page that has text only adds noise: on the ViT paper it read a
    watermark image and a stray "0" from a figure, which split a paragraph
    running across two pages, and it made the syllabus 5 times slower.
    """
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return _texts(_converter(formulas, ocr).convert(path))
    if path.suffix.lower() in UNPAGED_EXTENSIONS:
        is_html = path.suffix.lower() in HTML_EXTENSIONS
        source = _html_stream(path) if is_html else path
        text = document_text(_converter(formulas, "off").convert(source).document, furniture=not is_html)
        if is_html and (m := _SOURCE_URL_RE.search(path.read_bytes()[:4096])):
            text = f"Web page: {html.unescape(m.group(1).decode('utf-8', 'replace'))}\n\n{text}"
        return [text]
    if path.suffix.lower() in SLIDE_EXTENSIONS:
        return slide_texts(_converter(formulas, "off").convert(path).document)
    pages = _texts(_converter(formulas, "off").convert(path))
    missing = [n for n, chars in enumerate(text_layer_chars(path), start=1) if chars < MIN_TEXT_LAYER_CHARS]
    if missing and ocr != "off":
        # One OCR pass over the span of pages without text; only those pages are taken from it.
        ocr_pages = _texts(_converter(formulas, ocr).convert(path, page_range=(missing[0], missing[-1])))
        pages += [""] * (len(ocr_pages) - len(pages))
        for n in missing:
            pages[n - 1] = ocr_pages[n - 1]
    return pages


def text_layer_chars(path: Path) -> list[int]:
    """Characters in each PDF page's text layer (0 for a scanned page)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        return [len(page.get_textpage().get_text_range().strip()) for page in pdf]
    finally:
        pdf.close()


def _texts(result) -> list[str]:
    return page_texts(result.document, form_pages(result))


def slide_texts(doc) -> list[str]:
    """One markdown text per slide (index 0 = slide 1), each slide's speaker notes after it.

    The slide title is its heading, so the chunker names the section after
    it. Speaker notes often hold what the presenter says but the slide
    doesn't, so they are kept, marked "Speaker notes:".
    """
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
    from docling_core.types.doc import ContentLayer

    table_serializer, text_serializer = _serializers()

    def serialize(n: int, layer) -> str:
        params = MarkdownParams(pages={n}, layers={layer}, escape_html=False, escape_underscores=False,
                                image_placeholder="")
        return MarkdownDocSerializer(doc=doc, params=params, table_serializer=table_serializer(),
                                     text_serializer=text_serializer()).serialize().text.strip()

    texts = []
    for n in range(1, max(doc.pages, default=0) + 1):
        body, notes = serialize(n, ContentLayer.BODY), serialize(n, ContentLayer.NOTES)
        texts.append(body + (f"\n\nSpeaker notes: {notes}" if notes else ""))
    return texts


# Page parts that aren't the content, by tag, and by a word of their id or class ("mw-portlet-lang").
_CLUTTER_TAGS = ["nav", "aside", "form", "button", "script", "style", "noscript", "svg", "iframe",
                 "select", "input", "template", "dialog"]
_CLUTTER_WORDS = {
    "toc", "sidebar", "menu", "cookie", "cookies", "consent", "banner", "breadcrumb", "breadcrumbs",
    "share", "sharing", "social", "navbox", "navbar", "nav", "navigation", "lang", "language", "languages",
    "editsection", "toolbar", "catlinks", "related", "newsletter", "subscribe", "ad", "ads", "advert",
    "advertisement", "promo", "skip", "sitenotice", "noprint", "popup", "modal",
    # Footnote markers ("[1]", "[note 1]") and reference lists: the LLM cites with [n] markers itself.
    "reference", "references", "reflist",
}


def clean_html(data: bytes) -> bytes:
    """A web page reduced to its content before Docling reads it.

    Docling drops <nav>, <header> and <footer>, which was enough for a tidy
    test page, but a real Wikipedia article still brought its table of
    contents, a 17-language menu, "Edit links" and toggle buttons. Keep the
    page's <main> (or its <article>) when it has one, then remove clutter
    tags and elements whose id or class has a clutter word. Whole words
    only: "toc" removes "vector-toc", not "stock-price".
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "html.parser")
    root = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.find("article") or soup.body or soup
    for el in root.find_all(_CLUTTER_TAGS):
        el.decompose()
    for el in root.find_all(True):
        if el.decomposed:
            continue
        names = " ".join([el.get("id") or ""] + list(el.get("class") or []))
        hidden = el.has_attr("hidden") or el.get("aria-hidden") == "true"
        if hidden or _CLUTTER_WORDS & set(re.split(r"[^a-z0-9]+", names.lower())):
            el.decompose()
    title = soup.title.string if soup.title and soup.title.string else ""
    return f"<html><head><title>{html.escape(title)}</title></head><body>{root}</body></html>".encode()


def _html_stream(path: Path):
    import io

    from docling.datamodel.base_models import DocumentStream

    return DocumentStream(name=path.name, stream=io.BytesIO(clean_html(path.read_bytes())))


def document_text(doc, furniture: bool = True) -> str:
    """A document without pages (Word, HTML) as one markdown text.

    With `furniture`, its header and footer go first. A Word file stores them
    once, not on every page as a PDF does, and they often hold what someone
    may ask about (a document number, a version); before any heading, no
    section claims them. A web page's "furniture" is its menus, cookie banner
    and footer, so HTML leaves it out. Link targets are dropped: "[Shop](/shop)"
    is just "Shop" to a reader, and the address only adds noise to search.
    """
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
    from docling_core.types.doc import ContentLayer

    table_serializer, text_serializer = _serializers()

    def serialize(layer) -> str:
        params = MarkdownParams(layers={layer}, escape_html=False, escape_underscores=False, image_placeholder="",
                                include_hyperlinks=False)
        return MarkdownDocSerializer(doc=doc, params=params, table_serializer=table_serializer(),
                                     text_serializer=text_serializer()).serialize().text

    parts = ([serialize(ContentLayer.FURNITURE)] if furniture else []) + [serialize(ContentLayer.BODY)]
    text = "\n\n".join(t for t in parts if t.strip())
    return _SPACE_BEFORE_PUNCT_RE.sub("", text)  # inline links are joined with spaces: "See prices ."


def page_texts(doc, rows: dict[int, str] | None = None) -> list[str]:
    """One markdown text per page of a DoclingDocument (tables as rows, formulas checked).

    Index 0 is page 1. A document converted for a range of pages has "" for
    the pages before it. `rows` replaces whole pages (page number -> text):
    the form pages.
    """
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams

    table_serializer, text_serializer = _serializers()
    texts = []
    for n in range(1, max(doc.pages, default=0) + 1):
        if rows and n in rows:
            texts.append(rows[n])
            continue
        if n not in doc.pages:
            texts.append("")
            continue
        params = MarkdownParams(pages={n}, escape_html=False, escape_underscores=False, image_placeholder="")
        serializer = MarkdownDocSerializer(doc=doc, params=params, table_serializer=table_serializer(),
                                           text_serializer=text_serializer())
        texts.append(serializer.serialize().text)
    return texts


# -- forms -----------------------------------------------------------------------------


def form_pages(result) -> dict[int, str]:
    """Row-by-row text of each page that is mostly a form (page number -> text).

    Uses the text cells (OCR or text layer) of every layout region on the
    page, so the header and footer are kept. Checkboxes become "[x]" or "[ ]"
    where they are, as in Docling's markdown.
    """
    out = {}
    for page in result.pages:
        clusters = page.predictions.layout.clusters if page.predictions.layout else []
        area = page.size.width * page.size.height
        form = sum(c.bbox.area() for c in clusters if c.label.value == "form")
        if not area or form / area < FORM_PAGE_SHARE:
            continue
        boxes, seen = [], set()
        for cluster in clusters:
            if cluster.label.value in _CHECKBOXES:
                b = cluster.bbox.to_top_left_origin(page_height=page.size.height)
                boxes.append((b.t, b.b, b.l, b.r, _CHECKBOXES[cluster.label.value]))
            for cell in cluster.cells:
                if cell.index in seen:
                    continue
                seen.add(cell.index)
                b = cell.rect.to_bounding_box().to_top_left_origin(page_height=page.size.height)
                boxes.append((b.t, b.b, b.l, b.r, cell.text))
        out[page.page_no] = rows_text(boxes)  # 1-based, as in the DoclingDocument
    return out


_CHECKBOXES = {"checkbox_selected": "[x]", "checkbox_unselected": "[ ]"}


def rows_text(boxes: list[tuple[float, float, float, float, str]]) -> str:
    """Text boxes (top, bottom, left, right, text) as lines: visual rows, each read left to right.

    A box joins a row when its vertical centre is within half a box height of
    the row's. Within a row, boxes closer than the row's height are words of
    one phrase (one space); wider gaps separate fields (three spaces), which
    keeps a label and its value apart without a symbol a model might quote.
    """
    boxes = sorted(((t, b, l, r, " ".join(text.split())) for t, b, l, r, text in boxes if text.strip()),
                   key=lambda box: (box[0] + box[1]) / 2)
    rows: list[dict] = []
    for top, bottom, left, right, text in boxes:
        mid, height = (top + bottom) / 2, bottom - top
        for row in rows:
            if abs(mid - row["mid"]) < 0.5 * min(height, row["height"]):
                row["boxes"].append((left, right, text))
                break
        else:
            rows.append({"mid": mid, "height": height, "boxes": [(left, right, text)]})
    lines = []
    for row in rows:
        def order(box):  # a checkbox inside a text box's span ("(1) [x] the first…") goes first
            left, right, text = box
            if text in _CHECKBOXES.values():
                left = min([l - 0.1 for l, r, t in row["boxes"] if l <= left <= r and t != text] + [left])
            return left, text not in _CHECKBOXES.values()

        parts, prev_right = [], None
        for left, right, text in sorted(row["boxes"], key=order):
            if prev_right is not None:
                parts.append(" " if left - prev_right < row["height"] else "   ")
            parts.append(text)
            prev_right = right if prev_right is None else max(prev_right, right)
        lines.append("".join(parts))
    return "\n".join(lines)


# -- tables ----------------------------------------------------------------------------


def table_rows(item) -> str:
    """A TableItem as one "- header: value; header: value" line per body row.

    Header rows are the leading rows with column-header cells (row 0 if none
    is marked). Stacked headers join per column ("BLEU" over "EN-DE" becomes
    "BLEU EN-DE"). Empty cells are left out, and a cell spanning several
    columns is written once.
    """
    grid = item.data.grid
    if not grid:
        return ""
    n_head = 0
    while n_head < len(grid) - 1 and any(c.column_header for c in grid[n_head]):
        n_head += 1
    if n_head == 0 and len(grid) > 1:
        n_head = 1
    headers = []
    for j in range(len(grid[0])):
        parts: list[str] = []
        for r in range(n_head):
            text = " ".join((grid[r][j].text or "").split())
            if text and text not in parts:
                parts.append(text)
        headers.append(" ".join(parts))
    lines = []
    for row in grid[n_head:]:
        cells, seen = [], set()
        for j, cell in enumerate(row):
            text = " ".join((cell.text or "").split())
            span = (cell.start_row_offset_idx, cell.start_col_offset_idx)
            if not text or span in seen:
                continue
            seen.add(span)
            cells.append(f"{headers[j]}: {text}" if headers[j] and headers[j] != text else text)
        if cells:
            lines.append("- " + "; ".join(cells))
    return "\n".join(lines)


# -- formulas --------------------------------------------------------------------------


def formula_text(latex: str, pdf_text: str, inline: bool = False) -> str:
    """LaTeX when it agrees with the PDF's text, otherwise the PDF's text.

    When the two are identical, the LaTeX came from the file itself (a Word
    equation), not from a model reading an image, and is kept as it is.
    """
    exact = bool(latex.strip()) and latex.strip() == pdf_text.strip()
    latex = clean_latex(latex) if latex else ""
    if latex and (exact or same_symbols(latex, pdf_text)):
        return f"${latex}$" if inline else f"$${latex}$$"
    return " ".join(pdf_text.split()) if inline else pdf_text.strip()


def clean_latex(tex: str) -> str:
    r"""Docling's spaced-out LaTeX made compact, without the equation number.

    "A t t ( Q ) = \text {softmax} ( x ) & & ( 1 )" -> "Att(Q)=\text{softmax}(x)".
    Rows still separated by & or \\ get an aligned environment, which KaTeX needs.
    """
    tex = _EQ_NUMBER_RE.sub("", tex.strip()).strip()
    while tex.endswith("\\\\"):
        tex = tex[:-2].rstrip()
    out, prev = [], ""
    for tok in _LATEX_TOKEN_RE.findall(tex):
        if m := _LATEX_TEXT_RE.fullmatch(tok):
            tok = f"{tok[:tok.index('{')].rstrip()}{{{' '.join(m.group(1).split())}}}"
        if re.fullmatch(r"\\[A-Za-z]+", prev) and tok[0].isalpha():
            out.append(" ")  # "\cdot x", not "\cdotx"
        out.append(tok)
        prev = tok
    tex = "".join(out)
    if "&" in tex or "\\\\" in tex:
        tex = rf"\begin{{aligned}}{tex}\end{{aligned}}"
    return tex


def same_symbols(latex: str, pdf_text: str) -> bool:
    """Whether LaTeX and the PDF text hold the same ASCII letters and digits, in any order.

    Layout differs (a fraction's bar, sub- and superscripts), the symbols don't.
    A misread ("Attent" for "Attention") fails.
    """
    def symbols(text: str) -> Counter:
        return Counter(c for c in text if c.isascii() and c.isalnum())

    words = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}", " ", latex)  # environment names aren't content
    words = _LATEX_TEXT_RE.sub(r" \1 ", words)
    # A command, or a control symbol like "\\" (a line break: in "\\PE" the PE is content).
    words = re.sub(r"\\([A-Za-z]+|.)", lambda m: f" {m.group(1)} " if m.group(1) in _LATEX_WORDS else " ", words)
    plain = _EQ_NUMBER_RE.sub("", pdf_text.strip())
    return bool(symbols(plain)) and symbols(words) == symbols(plain)


# -- Docling plumbing ------------------------------------------------------------------


@lru_cache(maxsize=1)
def _serializers():
    """Subclasses of Docling's markdown serializers, defined on first use (Docling is optional)."""
    from docling_core.transforms.serializer.common import create_ser_result
    from docling_core.transforms.serializer.markdown import MarkdownTableSerializer, MarkdownTextSerializer
    from docling_core.types.doc import FormulaItem

    class RowTableSerializer(MarkdownTableSerializer):
        def serialize(self, *, item, doc_serializer, doc, **kwargs: Any):
            if kwargs.get("_nested_in_table"):
                return super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)
            parts = [doc_serializer.serialize_captions(item=item, **kwargs)]
            if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
                parts.append(create_ser_result(text=table_rows(item), span_source=item))
            parts.append(doc_serializer.serialize_footnotes(item=item, **kwargs))
            parts = [p for p in parts if p.text]
            return create_ser_result(text="\n\n".join(p.text for p in parts), span_source=parts)

    class TextSerializer(MarkdownTextSerializer):
        def serialize(self, *, item, doc_serializer, doc, is_inline_scope: bool = False, **kwargs: Any):
            if isinstance(item, FormulaItem):
                text = formula_text(item.text, item.orig or "", inline=is_inline_scope)
                return create_ser_result(text=text, span_source=item)
            return super().serialize(item=item, doc_serializer=doc_serializer, doc=doc,
                                     is_inline_scope=is_inline_scope, **kwargs)

    return RowTableSerializer, TextSerializer


@lru_cache(maxsize=4)
def _converter(formulas: str, ocr: str):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (OcrMacOptions, PdfPipelineOptions, RapidOcrOptions,
                                                    TableFormerMode)
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

    for name in ("docling", "docling_core", "docling_ibm_models", "docling_parse", "RapidOCR"):
        logging.getLogger(name).setLevel(logging.WARNING)
    opts = PdfPipelineOptions(do_ocr=ocr != "off", do_table_structure=True,
                              do_formula_enrichment=formulas == "latex")
    if ocr == "rapidocr":
        opts.ocr_options = RapidOcrOptions()
    elif ocr == "ocrmac":
        opts.ocr_options = OcrMacOptions(lang=["en-US"])
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE, InputFormat.DOCX, InputFormat.PPTX, InputFormat.HTML],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
                        InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)},
    )  # DOCX, PPTX and HTML use Docling's readers for them: no layout model needed, the file has structure


def _cache_path(data: bytes, formulas: str, ocr: str) -> Path:
    settings = f"v{PARSER_VERSION}|docling {importlib.metadata.version('docling')}|formulas {formulas}|ocr {ocr}"
    key = hashlib.sha256(data + b"\x00" + settings.encode()).hexdigest()[:32]
    return config.PARSE_CACHE_DIR / f"{key}.json"


def _write_cache(path: Path, pages: list[str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"pages": pages}, ensure_ascii=False))
        tmp.replace(path)
    except OSError as e:
        log.warning("could not write the parse cache (%s)", e)
