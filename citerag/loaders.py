"""PDF/TXT/MD files -> page records with source metadata. (Phase 1)

Every loader returns a list of `Page` records: one per PDF page, or a single
page 1 for text files. Text is cleaned here so later steps see plain prose:

- all files: unify line endings and odd Unicode (BOM, no-break spaces,
  ligatures like "ﬁ", zero-width characters), strip trailing spaces, and
  collapse runs of blank lines. Blank lines are kept — the chunker uses them
  as paragraph breaks.
- PDFs also get: running headers/footers and bare page numbers removed,
  hyphenated line breaks rejoined ("informa-\\ntion" -> "information"),
  and paragraph breaks inferred, because pypdf rarely emits blank lines.

Pages with no extractable text (blank or scanned) are skipped with a warning;
OCR is out of scope.

Try it:  python -m citerag.loaders data/
"""

from __future__ import annotations

import argparse
import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
from pypdf.errors import PdfReadError

log = logging.getLogger(__name__)
# pypdf logs a warning for every minor structural defect it repairs; too noisy.
logging.getLogger("pypdf").setLevel(logging.ERROR)

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf"}

EDGE_LINES = 2  # lines at the top and bottom of a page checked for headers/footers
REPEAT_MIN_PAGES = 3  # a header/footer must repeat on at least this many pages...
REPEAT_MIN_SHARE = 0.5  # ...and on at least this share of the document's pages


@dataclass
class Page:
    """One page of a source document. TXT/MD files are a single page 1."""

    text: str
    source: str  # file name (or relative path) shown in citations
    page: int  # 1-based


class LoaderError(Exception):
    """A file could not be read (unsupported, corrupt, or encrypted)."""


# -- loaders -----------------------------------------------------------------


def load_file(path: str | Path, source: str | None = None) -> list[Page]:
    """Load one supported file. `source` defaults to the file name."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path, source)
    if ext in TEXT_EXTENSIONS:
        return load_text(path, source)
    raise LoaderError(f"{path.name}: unsupported file type {ext or '(none)'}")


def load_text(path: str | Path, source: str | None = None) -> list[Page]:
    """Load a .txt/.md file as a single page 1."""
    path = Path(path)
    source = source or path.name
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise LoaderError(f"{source}: {e}") from e
    text = clean_text(_decode(raw, source))
    if not _has_text(text):
        log.warning("%s: file is empty, skipped", source)
        return []
    return [Page(text, source, 1)]


def load_pdf(path: str | Path, source: str | None = None) -> list[Page]:
    """Load a PDF, one `Page` per physical page (numbered as a PDF viewer shows them)."""
    path = Path(path)
    source = source or path.name
    try:
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise LoaderError(f"{source}: PDF is password-protected")
        raw_pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                raw_pages.append(page.extract_text() or "")
            except Exception as e:  # one broken page shouldn't lose the document
                log.warning("%s: could not read page %d (%s), skipped", source, i, e)
                raw_pages.append("")
    except LoaderError:
        raise
    except (PdfReadError, OSError, ValueError) as e:
        raise LoaderError(f"{source}: cannot read PDF ({e})") from e

    texts = clean_pdf_pages(raw_pages)
    pages = [Page(t, source, i) for i, t in enumerate(texts, start=1) if _has_text(t)]
    empty = [i for i, t in enumerate(texts, start=1) if not _has_text(t)]
    if not pages:
        log.warning("%s: no extractable text — probably scanned (OCR not supported)", source)
    elif empty:
        log.warning("%s: skipped %d page(s) with no text: %s", source, len(empty), _ranges(empty))
    return pages


def load_dir(path: str | Path, recursive: bool = True) -> list[Page]:
    """Load every supported file in a folder, in sorted order.

    Sources are paths relative to the folder (e.g. "reports/q3.pdf"), so files
    with the same name in different subfolders stay distinct. Unreadable
    files are skipped with a warning; hidden files and folders are ignored.
    """
    root = Path(path)
    if not root.is_dir():
        raise LoaderError(f"{root}: not a folder")
    files = root.rglob("*") if recursive else root.iterdir()
    pages: list[Page] = []
    for f in sorted(files):
        rel = f.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if not f.is_file() or f.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            pages.extend(load_file(f, rel.as_posix()))
        except LoaderError as e:
            log.warning("%s, skipped", e)
    return pages


def load_paths(paths: Iterable[str | Path]) -> list[Page]:
    """Load a mix of files and folders."""
    pages: list[Page] = []
    for p in map(Path, paths):
        pages.extend(load_dir(p) if p.is_dir() else load_file(p))
    return pages


# -- cleanup -----------------------------------------------------------------

_CHAR_MAP = str.maketrans(  # escapes, not literal characters: editors mangle invisibles
    {
        "\r": "\n", "\u2028": "\n", "\u2029": "\n\n", "\f": "\n\n",
        "\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u3000": " ",
        "\u200b": None, "\u200c": None, "\u200d": None, "\u2060": None, "\ufeff": None,
        "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
        "\ufb05": "st", "\ufb06": "st",
        # Bullets drawn with the Symbol/Wingdings fonts (Word exports) arrive as
        # Private Use Area code points; the chunker recognises "\u2022" as a list item.
        "\uf0b7": "\u2022", "\uf0a7": "\u2022", "\uf0a8": "\u2022", "\uf06c": "\u2022",
        "\uf06e": "\u2022", "\uf076": "\u2022", "\uf0d8": "\u2022", "\uf0fc": "\u2022",
    }
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_SPACE_RUN_RE = re.compile(r"[ \t]{2,}")
# "informa-\ntion" -> "information"; also a soft hyphen at a line end.
_HYPHEN_BREAK_RE = re.compile(r"(?<=[A-Za-z])(?:-|\u00ad)\n[ \t]*(?=[a-z])")
_PAGE_NUMBER_RE = re.compile(
    r"(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?|[-–—]\s*\d{1,4}\s*[-–—]"
    r"|(?=[ivxlc])(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})",  # roman numerals up to 99
    re.IGNORECASE,
)
_SENTENCE_END = ".!?:"
_CLOSERS = "\"'”’)]"
# A line ending like this is mid-sentence, so the next line always continues it.
_DANGLING_RE = re.compile(
    r"(?:[,;/&(–-]|\b(?:a|an|the|and|or|nor|but|of|to|in|on|at|by|for|from|with|as|into"
    r"|onto|than|that|which|who|whose|via|vs|versus|between|including|such|its|their|our))$",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Normalize characters and whitespace; keeps blank lines (paragraph breaks)."""
    text = text.replace("\r\n", "\n").translate(_CHAR_MAP)
    text = _CONTROL_RE.sub("", text)
    text = _TRAILING_WS_RE.sub("\n", text + "\n")
    return _BLANK_RUN_RE.sub("\n\n", text).strip()


def clean_pdf_pages(raw_pages: list[str]) -> list[str]:
    """Clean the extracted text of all pages of one PDF."""
    page_lines = [
        [line.strip() for line in _SPACE_RUN_RE.sub(" ", clean_text(t)).split("\n")]
        for t in raw_pages
    ]
    page_lines = _drop_headers_footers(page_lines)
    texts = []
    for lines in page_lines:
        text = "\n".join(lines)
        if "\n\n" not in text:
            text = _infer_paragraphs(lines)
        text = _HYPHEN_BREAK_RE.sub("", text)
        texts.append(_BLANK_RUN_RE.sub("\n\n", text).strip())
    return texts


def _drop_headers_footers(page_lines: list[list[str]]) -> list[list[str]]:
    """Remove bare page numbers and lines repeated at the top/bottom of many pages.

    Digits are ignored when comparing, so "Report 2024 — page 3" and
    "Report 2024 — page 4" count as the same running header.
    """

    def edges(lines: list[str]) -> list[int]:
        idx = [i for i, line in enumerate(lines) if line]
        return sorted(set(idx[:EDGE_LINES] + idx[-EDGE_LINES:]))

    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line.lower())

    counts = Counter(k for lines in page_lines for k in {key(lines[i]) for i in edges(lines)})
    threshold = max(REPEAT_MIN_PAGES, REPEAT_MIN_SHARE * len(page_lines))
    repeated = {k for k, n in counts.items() if n >= threshold}

    cleaned = []
    for lines in page_lines:
        drop = {
            i
            for i in edges(lines)
            if key(lines[i]) in repeated or _PAGE_NUMBER_RE.fullmatch(lines[i])
        }
        cleaned.append([line for i, line in enumerate(lines) if i not in drop])
    return cleaned


def _infer_paragraphs(lines: list[str]) -> str:
    """Rebuild paragraph breaks from line shapes (pypdf gives one line per row).

    A line that ends a sentence while stopping well short of the page's text
    width ends a paragraph. A short line without end punctuation (a likely
    heading or list entry) gets a paragraph of its own. A line is always
    continued when the next one starts in lowercase or when it ends on a word
    or mark that can't end a sentence ("and", "of", ","), e.g. a table cell
    "Understand XHTML and" / "HTML5" — otherwise "HTML5" would look like a heading.
    """
    lines = [line for line in lines if line]
    if len(lines) < 3:
        return "\n".join(lines)
    lengths = sorted(len(line) for line in lines)
    width = lengths[int(0.9 * (len(lengths) - 1))]  # robust "full line" length

    def is_short(line: str, share: float) -> bool:
        return len(line) < share * width

    def is_standalone(line: str) -> bool:
        return is_short(line, 0.6) and line[-1] not in _SENTENCE_END + ",;-"

    out = [lines[0]]
    for prev, line in zip(lines, lines[1:]):
        continues = line[0].islower() or bool(_DANGLING_RE.search(prev))
        ends_sentence = prev.rstrip(_CLOSERS)[-1:] in _SENTENCE_END
        para_end = ends_sentence and is_short(prev, 0.85)
        brk = not continues and (para_end or is_standalone(prev) or is_standalone(line))
        out.append("\n\n" if brk else "\n")
        out.append(line)
    return "".join(out)


# -- helpers -----------------------------------------------------------------


def _decode(raw: bytes, source: str) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        log.warning("%s: not valid UTF-8, decoded as Windows-1252", source)
        return raw.decode("cp1252", errors="replace")


def _has_text(text: str) -> bool:
    return any(c.isalnum() for c in text)


def _ranges(nums: list[int]) -> str:
    """[1, 2, 3, 7] -> "1-3, 7"."""
    parts, start = [], nums[0]
    for a, b in zip(nums, nums[1:] + [None]):
        if b != a + 1:
            parts.append(f"{start}-{a}" if a != start else str(a))
            start = b
    return ", ".join(parts)


# -- CLI ---------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Preview what the loaders extract.")
    parser.add_argument("paths", nargs="+", type=Path, help="files or folders")
    parser.add_argument("--show", type=int, default=1, help="pages of text to print")
    parser.add_argument("--chars", type=int, default=800, help="characters per page shown")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="warning: %(message)s")

    pages = load_paths(args.paths)
    if not pages:
        print("No text found.")
        return
    by_source = Counter(p.source for p in pages)
    for source, n in by_source.items():
        chars = sum(len(p.text) for p in pages if p.source == source)
        print(f"{source}: {n} page(s), {chars:,} chars")
    print(f"total: {len(pages)} page(s) from {len(by_source)} file(s), "
          f"median {statistics.median(len(p.text) for p in pages):.0f} chars/page")
    for p in pages[: args.show]:
        print(f"\n--- {p.source} p{p.page}")
        print(p.text[: args.chars] + (" …" if len(p.text) > args.chars else ""))


if __name__ == "__main__":
    _main()
