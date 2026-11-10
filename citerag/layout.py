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

Parsing takes seconds per page plus a model load, so results are cached on
disk by file content and parser settings. Re-scanning a folder only parses
new or changed files.
"""

from __future__ import annotations

import hashlib
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
PARSER_VERSION = 1
FORMULA_MODES = ("text", "latex")

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


def parse_pdf(path: str | Path, formulas: str | None = None) -> list[str]:
    """The text of each physical page (index 0 = page 1), from cache when possible."""
    formulas = formulas or config.PDF_FORMULAS
    if formulas not in FORMULA_MODES:
        raise ValueError(f"CITERAG_PDF_FORMULAS must be one of {FORMULA_MODES}, not {formulas!r}")
    data = Path(path).read_bytes()
    cache = _cache_path(data, formulas)
    if cache.is_file():
        try:
            return json.loads(cache.read_text())["pages"]
        except (OSError, ValueError, KeyError):
            log.warning("%s: unreadable parse cache, parsing again", Path(path).name)
    with _lock:
        try:
            doc = _converter(formulas).convert(Path(path)).document
        except Exception as e:  # Docling raises many types; the loader falls back to pypdf
            raise LayoutError(str(e) or type(e).__name__) from e
    pages = page_texts(doc)
    _write_cache(cache, pages)
    return pages


def page_texts(doc) -> list[str]:
    """One markdown text per page of a DoclingDocument (tables as rows, formulas checked)."""
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams

    table_serializer, text_serializer = _serializers()
    texts = []
    for n in range(1, doc.num_pages() + 1):
        params = MarkdownParams(pages={n}, escape_html=False, escape_underscores=False, image_placeholder="")
        serializer = MarkdownDocSerializer(doc=doc, params=params, table_serializer=table_serializer(),
                                           text_serializer=text_serializer())
        texts.append(serializer.serialize().text)
    return texts


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
    """LaTeX when it agrees with the PDF's text, otherwise the PDF's text."""
    latex = clean_latex(latex) if latex else ""
    if latex and same_symbols(latex, pdf_text):
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


@lru_cache(maxsize=2)
def _converter(formulas: str):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    for name in ("docling", "docling_core", "docling_ibm_models", "docling_parse"):
        logging.getLogger(name).setLevel(logging.WARNING)
    opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True,
                              do_formula_enrichment=formulas == "latex")
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def _cache_path(data: bytes, formulas: str) -> Path:
    settings = f"v{PARSER_VERSION}|docling {importlib.metadata.version('docling')}|formulas {formulas}|ocr off"
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
