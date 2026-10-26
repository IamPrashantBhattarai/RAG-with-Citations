"""Split page records into overlapping chunks with stable IDs. (Phase 2)

Strategy: structure-aware sentence packing.

1. Join a document's pages into one text, so a paragraph that crosses a page
   break stays together; remember where each page starts for citations.
2. Split into paragraphs, detect headings, and split paragraphs into
   sentences (an over-long sentence falls back to word windows).
3. Greedily pack whole sentences into chunks of up to CHUNK_SIZE tokens,
   preferring to break at headings and paragraph ends.
4. When a cut has to fall inside a paragraph, repeat the last sentences
   (up to CHUNK_OVERLAP tokens) at the start of the next chunk.
5. Prefix each chunk's embedding text with "title > section", so the
   vector also knows what the passage is about.

All sizes are counted with the embedding model's tokenizer, so a chunk is
never silently truncated at embedding time. Chunk text is an exact slice of
the source, so quoted citations can be verified against it later.

Try it:  python -m citerag.chunker notes.md report.pdf --size 384 --overlap 64
"""

from __future__ import annotations

import argparse
import bisect
import re
import statistics
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from typing import Callable, Iterable

from citerag import config
from citerag.loaders import Page, load_paths

TokenCounter = Callable[[list[str]], list[int]]

HEADER_MAX_CHARS = 120
HEADER_TOKEN_BUDGET = 48  # room reserved for "title > section" in embed_text
SPECIAL_TOKENS = 2  # [CLS] + [SEP] added by the embedding model

_PARA_RE = re.compile(r"\n[ \t]*\n+")
# Sentence end, or a newline that starts a list item ("- x", "• x", "3. x", "a) x").
_BOUNDARY_RE = re.compile(
    r"[.!?]+[\"'”’)\]]*\s+"
    r"|\n(?=[ \t]*(?:[-*•▪◦]|\d{1,3}[.)]|[a-z][.)])[ \t]+)"
)
_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "cf", "al", "approx", "mr", "mrs", "ms", "dr",
    "prof", "sr", "jr", "st", "no", "nos", "fig", "figs", "eq", "vol", "pp",
    "ch", "sec", "dept", "inc", "ltd", "co", "corp", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_SENTENCE_START = "\"'“‘([-*•"
_MD_HEADING_RE = re.compile(r"#{1,6}[ \t]+(.+)")
_UNDERLINE_RE = re.compile(r"(?m)^[ \t]*(?:={3,}|-{3,})[ \t]*$")
_NUM_HEADING_RE = re.compile(
    r"(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.|(?:chapter|section|part|appendix)\s+\w+)\s+\S.*",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    id: str
    text: str  # exact source text — shown and verified in citations
    embed_text: str  # "title > section" header + whitespace-normalized text
    source: str
    page_start: int
    page_end: int
    section: str | None
    start_char: int  # offsets into the document's joined text
    end_char: int
    n_tokens: int


@dataclass
class _Unit:
    """A sentence (or word window of an over-long sentence)."""

    start: int
    end: int
    section: str | None
    para_start: bool = False
    heading: bool = False
    tokens: int = 0
    para_tokens: int = 0  # whole-paragraph size; set on the paragraph's first unit


@lru_cache(maxsize=4)
def hf_token_counter(model_name: str = config.EMBED_MODEL) -> TokenCounter:
    """Count tokens exactly as the embedding model will see them."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 10**9  # we only count; silence length warnings

    def count(texts: list[str]) -> list[int]:
        if not texts:
            return []
        ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [len(x) for x in ids]

    return count


class Chunker:
    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        overlap: int = config.CHUNK_OVERLAP,
        min_tokens: int = config.CHUNK_MIN,
        count_tokens: TokenCounter | None = None,
        embed_max_tokens: int = config.EMBED_MAX_TOKENS,
    ):
        self.max_tokens = embed_max_tokens - SPECIAL_TOKENS - HEADER_TOKEN_BUDGET
        if not 0 < chunk_size <= self.max_tokens:
            raise ValueError(
                f"chunk_size must be 1..{self.max_tokens} for a "
                f"{embed_max_tokens}-token embedding model, got {chunk_size}"
            )
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        if not 0 <= min_tokens < chunk_size:
            raise ValueError("min_tokens must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_tokens = min_tokens
        self._count = count_tokens

    @property
    def count_tokens(self) -> TokenCounter:
        if self._count is None:
            self._count = hf_token_counter()
        return self._count

    def chunk_pages(self, pages: Iterable[Page]) -> list[Chunk]:
        """Chunk pages from any number of documents (grouped by source, in order)."""
        chunks: list[Chunk] = []
        for _, doc_pages in groupby(pages, key=lambda p: p.source):
            chunks.extend(self.chunk_document(list(doc_pages)))
        return chunks

    def chunk_document(self, pages: list[Page]) -> list[Chunk]:
        """Chunk the pages of a single document."""
        pages = [p for p in sorted(pages, key=lambda p: p.page) if p.text.strip()]
        if not pages:
            return []
        source = pages[0].source
        text, page_offsets = _join_pages(pages)
        units = self._units(text)
        groups = self._pack(units)

        title = _title(source)
        slug = _slug(source)
        chunks = []
        for i, group in enumerate(groups):
            start, end = group[0].start, group[-1].end
            p_start = pages[bisect.bisect_right(page_offsets, start) - 1].page
            p_end = pages[bisect.bisect_right(page_offsets, end - 1) - 1].page
            section = group[0].section
            body = text[start:end]
            header = f"{title} > {section}" if section else title
            chunks.append(
                Chunk(
                    id=f"{slug}-p{p_start}-c{i}",
                    text=body,
                    embed_text=f"{header[:HEADER_MAX_CHARS]}\n\n{_normalize(body)}",
                    source=source,
                    page_start=p_start,
                    page_end=p_end,
                    section=section,
                    start_char=start,
                    end_char=end,
                    n_tokens=sum(u.tokens for u in group),
                )
            )
        return chunks

    # -- splitting ---------------------------------------------------------

    def _units(self, text: str) -> list[_Unit]:
        units: list[_Unit] = []
        section: str | None = None
        for p_start, p_end in _paragraph_spans(text):
            for b_start, b_end, heading in _blocks(text, p_start, p_end):
                if heading is not None:
                    section = heading
                    units.append(_Unit(b_start, b_end, section, True, True))
                    continue
                spans = _sentence_spans(text, b_start, b_end)
                for j, (s, e) in enumerate(spans):
                    units.append(_Unit(s, e, section, para_start=j == 0))

        counts = self.count_tokens([text[u.start : u.end] for u in units])
        sized: list[_Unit] = []
        for unit, n in zip(units, counts):
            unit.tokens = n
            if n > self.chunk_size:
                sized.extend(self._split_long(text, unit))
            else:
                sized.append(unit)

        para_head = None
        for unit in sized:
            if unit.para_start:
                para_head = unit
            if para_head is not None:
                para_head.para_tokens += unit.tokens
        return sized

    def _split_long(self, text: str, unit: _Unit) -> list[_Unit]:
        """Break a sentence longer than chunk_size into word windows."""
        words = list(re.finditer(r"\S+", text[unit.start : unit.end]))
        counts = self.count_tokens([w.group() for w in words])
        pieces: list[_Unit] = []
        w_start, n = 0, 0
        for k, c in enumerate(counts + [None]):
            if c is None or (n + c > self.chunk_size and k > w_start):
                pieces.append(
                    _Unit(
                        unit.start + words[w_start].start(),
                        unit.start + words[k - 1].end(),
                        unit.section,
                        para_start=unit.para_start and not pieces,
                        tokens=n,
                    )
                )
                w_start, n = k, 0
            if c is not None:
                n += c
        return pieces

    # -- packing -----------------------------------------------------------

    def _pack(self, units: list[_Unit]) -> list[list[_Unit]]:
        groups: list[tuple[list[_Unit], int]] = []  # (units, n carried from previous)
        cur: list[_Unit] = []
        carried = carried_tokens = tokens = 0

        def reset(keep: list[_Unit] = ()) -> None:
            nonlocal cur, carried, carried_tokens, tokens
            cur = list(keep)
            carried = len(cur)
            carried_tokens = tokens = sum(u.tokens for u in cur)

        def emit(overlap: bool) -> None:
            if len(cur) > carried:
                groups.append((cur, carried))
            reset(self._overlap_tail(cur) if overlap and len(cur) > carried else ())

        for u in units:
            fresh = tokens - carried_tokens
            if u.heading and fresh >= self.min_tokens:
                emit(overlap=False)  # new section starts a new chunk
            elif u.heading and len(cur) == carried:
                reset()  # don't drag the previous section's overlap into a new one
            elif tokens + u.tokens > self.chunk_size:
                emit(overlap=not u.para_start)  # overlap only repairs mid-paragraph cuts
                if tokens + u.tokens > self.chunk_size:
                    reset()
            elif (
                u.para_start
                and fresh >= self.min_tokens
                and tokens + u.para_tokens > self.chunk_size
            ):
                emit(overlap=False)  # next paragraph won't fit: break cleanly here
            cur.append(u)
            tokens += u.tokens
        emit(overlap=False)

        # Fold a tiny trailing chunk into its neighbour instead of embedding a scrap.
        if len(groups) >= 2:
            last, last_carried = groups[-1]
            tail = last[last_carried:]
            tail_tokens = sum(u.tokens for u in tail)
            prev, prev_carried = groups[-2]
            if (
                tail_tokens < self.min_tokens
                and not tail[0].heading
                and sum(u.tokens for u in prev) + tail_tokens <= self.max_tokens
            ):
                groups[-2:] = [(prev + tail, prev_carried)]
        return [g for g, _ in groups]

    def _overlap_tail(self, units: list[_Unit]) -> list[_Unit]:
        tail: list[_Unit] = []
        n = 0
        for u in reversed(units[1:]):  # never carry the whole chunk
            if u.heading or n + u.tokens > self.overlap:
                break
            tail.insert(0, u)
            n += u.tokens
        return tail


def chunk_pages(pages: Iterable[Page]) -> list[Chunk]:
    """Chunk pages with the default settings from config."""
    return Chunker().chunk_pages(pages)


# -- helpers ---------------------------------------------------------------


def _join_pages(pages: list[Page]) -> tuple[str, list[int]]:
    """Join pages; a page ending mid-sentence continues with a space, not a break."""
    parts: list[str] = []
    offsets: list[int] = []
    pos = 0
    for i, page in enumerate(pages):
        body = page.text.strip()
        if i:
            prev = parts[-1].rstrip()
            sep = "\n\n" if prev[-1:] in ".!?:\"”)]" else " "
            parts.append(sep)
            pos += len(sep)
        offsets.append(pos)
        parts.append(body)
        pos += len(body)
    return "".join(parts), offsets


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for m in _PARA_RE.finditer(text):
        spans.append(_trim(text, pos, m.start()))
        pos = m.end()
    spans.append(_trim(text, pos, len(text)))
    return [(s, e) for s, e in spans if s < e]


def _blocks(text: str, start: int, end: int) -> list[tuple[int, int, str | None]]:
    """A paragraph, or a heading line plus the paragraph it introduces."""
    para = text[start:end]
    heading = _heading(para)
    if heading is not None:
        return [(start, end, heading)]
    nl = para.find("\n")
    if nl != -1:
        first = para[:nl].strip()
        nl2 = para.find("\n", nl + 1)
        head_len = nl2 if nl2 != -1 else len(para)
        underline = para[nl + 1 : head_len].strip()
        if len(underline) >= 3 and set(underline) in ({"="}, {"-"}) and len(first) <= 100:
            blocks = [(*_trim(text, start, start + head_len), first)]  # setext heading
            rest_start, rest_end = _trim(text, start + head_len, end)
            if rest_start < rest_end:
                blocks.append((rest_start, rest_end, None))
            return blocks
        # Only unambiguous styles here; "1. Foo" on a first line is usually a list item.
        if _MD_HEADING_RE.fullmatch(first) or _is_caps_heading(first):
            h_start, h_end = _trim(text, start, start + nl)
            rest_start, rest_end = _trim(text, start + nl, end)
            return [(h_start, h_end, _heading(first)), (rest_start, rest_end, None)]
    return [(start, end, None)]


def _heading(line: str) -> str | None:
    line = line.strip()
    if "\n" in line or not line or len(line) > 100:
        return None
    m = _MD_HEADING_RE.fullmatch(line)
    if m:
        return m.group(1).strip(" #")
    if line[-1] in ".,;!?":
        return None
    if _NUM_HEADING_RE.fullmatch(line) and len(line.split()) <= 12:
        return line
    if _is_caps_heading(line):
        return line
    return None


def _is_caps_heading(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return (
        len(letters) >= 3
        and all(c.isupper() for c in letters)
        and len(line.split()) <= 10
        and line[-1:] not in ".,;!?"
    )


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    s = start
    for m in _BOUNDARY_RE.finditer(text, start, end):
        cut = m.end()
        if cut >= end:
            break
        punct = m.group()
        if punct[0] != "\n":
            if not (text[cut].isupper() or text[cut].isdigit() or text[cut] in _SENTENCE_START):
                continue
            if punct[0] == ".":
                before = text[s : m.start()].split()
                word = before[-1].lower().lstrip("\"'“‘(") if before else ""
                if word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
                    continue  # "e.g. Foo", "Dr. Smith", "J. Smith"
        spans.append(_trim(text, s, m.start() + len(punct.rstrip())))
        s = cut
    spans.append(_trim(text, s, end))
    return [(a, b) for a, b in spans if a < b]


def _normalize(text: str) -> str:
    """Collapse whitespace and drop heading underlines for the embedding text."""
    return " ".join(_UNDERLINE_RE.sub("", text).split())


def _title(source: str) -> str:
    return re.sub(r"[_\-]+", " ", Path(source).stem).strip() or source


def _slug(source: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-") or "doc"


# -- CLI ---------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Preview how files are chunked.")
    parser.add_argument("files", nargs="+", type=Path, help="PDF/TXT/MD files or folders")
    parser.add_argument("--size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--min", type=int, default=config.CHUNK_MIN)
    parser.add_argument("--show", type=int, default=3, help="chunks to print")
    args = parser.parse_args()

    chunker = Chunker(args.size, args.overlap, args.min)
    pages = load_paths(args.files)
    chunks = chunker.chunk_pages(pages)
    if not chunks:
        print("No text found.")
        return
    sizes = [c.n_tokens for c in chunks]
    print(
        f"{len(chunks)} chunks | tokens min {min(sizes)} / "
        f"median {statistics.median(sizes):.0f} / max {max(sizes)}"
    )
    for c in chunks[: args.show]:
        print(f"\n--- {c.id}  ({c.n_tokens} tok, section: {c.section})")
        print(c.embed_text)


if __name__ == "__main__":
    _main()
