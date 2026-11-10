"""Helpers for the Streamlit app (app.py). (Phase 7)

They turn an `Answer` into markdown that is safe to render, highlight quotes
in their passages, save uploads, and cut cited pages out of PDFs. None of
them import Streamlit, so they are unit-tested; app.py only does the layout.

Safety: answer text comes from an LLM that has read the user's documents, and
a document could contain instructions ("output this HTML…"). Everything from
the LLM or a document is HTML-escaped before it is rendered with
`unsafe_allow_html`, and markdown images are removed (an image URL would be
fetched as soon as the answer is shown).

Formulas: math the LLM writes as $…$ or $$…$$ is rendered by Streamlit's
KaTeX (which runs untrusted: no \\href, \\url or \\html* commands). Any other
"$" (prices: "$5 and $10") is escaped. Inside a formula "<" and ">" become
\\lt and \\gt, so a formula can't hold an HTML tag even if Streamlit split
the text differently from us.
"""

from __future__ import annotations

import html
import io
import os
import re
from pathlib import Path

from citerag.generator import Answer
from citerag.loaders import SUPPORTED_EXTENSIONS
from citerag.store import IndexReport
from citerag.validator import Validation, display_offsets

_MARKER_RE = re.compile(r"\[(\d+)\]")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LEAD_RE = re.compile(r"[ \t]*(?:[-*•]|\d{1,3}[.)]|#{1,6})[ \t]+")  # list bullet / heading marker
_TRAILING_MARKERS_RE = re.compile(r"(?:[ \t]*\[\d+\])+")
_MD_SPECIAL_RE = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")
_DOLLAR_RE = re.compile(r"\\*\$")  # "$", or "\$" the LLM escaped itself
# $$…$$ (may span lines, not a blank one), or $…$ on one line with pandoc's rules:
# no space just inside the dollars and no digit right after, so "$5 and $10" isn't math.
_MATH_RE = re.compile(
    r"(?<![\\$])\$\$(?P<display>(?:[^$\n]|\n(?![ \t]*\n))+?)\$\$(?!\$)"
    r"|(?<![\\$])\$(?P<inline>[^\s$][^$\n]*?(?<![\s\\]))\$(?![\d$])"
)
WARN_STYLE = "background: rgba(250, 204, 21, 0.3); border-radius: 3px; padding: 0 2px"
QUOTE_STYLE = "background: rgba(250, 204, 21, 0.45); color: inherit; padding: 0 1px"


def answer_markdown(answer: Answer) -> str:
    """The answer as markdown, with warned statements highlighted.

    A warned statement gets a yellow background with its reasons as a
    tooltip. Markers become superscripts and formulas are rendered.
    """
    text = answer.text
    math = _math_spans(text)
    warned = []
    v = answer.validation
    if v is not None and v.warnings:
        pos = display_offsets(text, "".join(s.text for s in answer.segments))
        for c in v.warnings:
            start, end = pos[c.start], pos[c.end]
            if lead := _LEAD_RE.match(text, start, end):
                start = lead.end()  # keep "- " outside, or the list item breaks
            if markers := _TRAILING_MARKERS_RE.match(text, end):
                end = markers.end()
            for m_start, m_end, _ in math:  # never cut a formula in two
                if m_start < start < m_end:
                    start = m_start
                if m_start < end < m_end:
                    end = m_end
            warned.append((start, end, c.issues))
    out, i = [], 0
    for start, end, issues in sorted(warned):
        if start < i:
            continue
        tip = html.escape("; ".join(issues), quote=True)
        out.append(_render(text, i, start, math, _escape))
        out.append(f'<span style="{WARN_STYLE}" title="{tip}">{_render(text, start, end, math, _escape)}</span>')
        i = end
    out.append(_render(text, i, len(text), math, _escape))
    return "".join(out)


def streaming_markdown(text: str) -> str:
    """Partial answer while it streams (rendered without HTML).

    A formula shows as LaTeX source until its closing "$" arrives.
    """
    return _render(text, 0, len(text), _math_spans(text),
                   lambda s: _DOLLAR_RE.sub(r"\\$", _IMAGE_RE.sub(r"\1", s)))


def passage_html(text: str, start: int, end: int) -> str:
    """A passage with text[start:end] highlighted, as a single-line HTML block.

    Line breaks become <br> so the block has no blank line (which would end
    an HTML block and let markdown parse the rest).
    """
    def esc(s: str) -> str:
        return html.escape(s).replace("\n", "<br>")

    body = esc(text[:start]) + f'<mark style="{QUOTE_STYLE}">{esc(text[start:end])}</mark>' + esc(text[end:])
    return f'<div style="font-size: 0.9em; line-height: 1.5">{body}</div>'


def check_lines(v: Validation) -> list[str]:
    """One markdown line per checked statement (✓ or ✗, reasons, NLI score)."""
    lines = []
    for c in v.claims:
        marks = "".join(f"[{n}]" for n in c.citations)
        score = f" · support {c.entailment:.2f}" if c.entailment is not None else ""
        line = f"{'✓' if c.status == 'ok' else ':orange[✗]'} {escape_md(c.text)} {escape_md(marks)}"
        if c.issues:
            line += f"  \n  ↳ *{escape_md('; '.join(c.issues))}*{score}"
        elif score:
            line += f"  \n  ↳ *{score.lstrip(' ·')}*"
        lines.append(line)
    for c in v.removed:
        lines.append(f"~~{escape_md(c.text)}~~  \n  ↳ *removed by strict mode: {escape_md('; '.join(c.issues))}*")
    return lines


def summary_line(answer: Answer) -> str | None:
    v = answer.validation
    if v is None or not (v.claims or v.removed or v.invalid_markers):
        return None
    text = v.summary()
    return text + (". Unsupported statements are highlighted." if v.warnings else ".")


def escape_md(text: str) -> str:
    """Show text literally in Streamlit markdown."""
    return _MD_SPECIAL_RE.sub(r"\\\1", html.escape(text, quote=False))


def _escape(text: str) -> str:
    """LLM markdown (no formulas in it) kept as markdown, but no raw HTML, no images, no "$" math."""
    text = _DOLLAR_RE.sub(r"\\$", _IMAGE_RE.sub(r"\1", html.escape(text, quote=False)))
    return _MARKER_RE.sub(r"<sup>[\1]</sup>", text)


def _math_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, markdown) for each formula in `text`, in order.

    $$…$$ on a line of its own becomes a display block, kept at the line's
    indent so it stays inside a list item. Anywhere else it is inline math in
    \\displaystyle (Streamlit draws $$…$$ in running text small, like $…$).
    """
    spans = []
    for m in _MATH_RE.finditer(text):
        if m["inline"] is not None:
            spans.append((m.start(), m.end(), f"${_safe_tex(m['inline'])}$"))
            continue
        tex = _safe_tex(m["display"].strip())
        if not tex or tex.endswith("\\"):
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        indent = text[line_start:m.start()]
        rest = text[m.end():len(text) if line_end < 0 else line_end]
        if not indent.strip() and not rest.strip():
            body = "\n".join(indent + line.strip() for line in tex.split("\n"))
            spans.append((m.start(), m.end(), f"$$\n{body}\n{indent}$$"))
        else:
            spans.append((m.start(), m.end(), rf"$\displaystyle {' '.join(tex.split())}$"))
    return spans


def _safe_tex(tex: str) -> str:
    return tex.replace("<", r"\lt ").replace(">", r"\gt ")


def _render(text: str, start: int, end: int, math: list[tuple[int, int, str]], escape) -> str:
    """text[start:end] with its formulas as markdown math and `escape` applied to the rest.

    start and end never fall inside a formula.
    """
    out, i = [], start
    for m_start, m_end, md in math:
        if m_start >= start and m_end <= end:
            out.append(escape(text[i:m_start]))
            out.append(md)
            i = m_end
    out.append(escape(text[i:end]))
    return "".join(out)


# -- documents ---------------------------------------------------------------------


def save_upload(name: str, data: bytes, folder: Path) -> Path:
    """Save an uploaded file into `folder` under its bare file name.

    Any directory part of the name is dropped, so "../../x.pdf" can't write
    outside the folder. An existing file with the same name is replaced.
    """
    filename = Path(name.replace("\\", "/")).name
    if not filename or filename.startswith("."):
        raise ValueError(f"invalid file name: {name!r}")
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"{filename}: only PDF, TXT and MD files are supported")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_bytes(data)
    return path


def report_text(report: IndexReport) -> str:
    parts = []
    for label, files in (("Added", report.added), ("Updated", report.updated),
                         ("Unchanged", report.unchanged), ("Removed", report.removed)):
        if files:
            parts.append(f"{label}: {', '.join(files)}")
    if report.chunks:
        parts.append(f"{report.chunks} chunk{'s' * (report.chunks != 1)} embedded")
    return "; ".join(parts) or "Nothing to index"


def source_pdf(source: str, folder: Path) -> Path | None:
    """The cited PDF in `folder`, if it is there (and really inside it)."""
    root = folder.resolve()
    path = (root / source).resolve()
    if path.suffix.lower() != ".pdf" or not path.is_relative_to(root) or not path.is_file():
        return None
    return path


def page_pdf(path: Path, first: int, last: int) -> bytes:
    """Pages first..last (1-based, as cited) of a PDF, as a new PDF."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(path)
    writer = PdfWriter()
    for n in range(max(first, 1), min(last, len(reader.pages)) + 1):
        writer.add_page(reader.pages[n - 1])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def has_key(provider: str) -> bool:
    """Whether the provider's API key is set (without reading it out)."""
    names = ["ANTHROPIC_API_KEY"] if provider == "anthropic" else ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
    return any(os.getenv(n, "") not in ("", "...", "sk-ant-...") for n in names)
