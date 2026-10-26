"""Parse and validate [n] citations against retrieved chunks. (Phase 5 parsing; Phase 6 validation)

Models without a citations API (Gemini here) are given numbered excerpts and
asked to put markers like [2] after each claim. `InlineCitations` turns that
text into the same `Segment`/`Citation` objects the Claude backend produces:

- it reads markers while the answer streams (a marker can be split across
  chunks, e.g. "[" then "2]"), so the live text and the final text match;
- it renumbers citations by first appearance, like the Claude backend;
- for each cited claim it finds the shortest passage of the excerpt (a
  sentence, two adjacent sentences, or a line of a table) that shares about
  as many words with the claim as any, so every citation still carries an
  exact quote with character offsets. `Citation.support` records how much
  of the claim that quote covers (0..1);
- text before the claim's own sentence stays an uncited segment, and
  markers pointing at excerpts that don't exist are dropped and recorded.
"""

from __future__ import annotations

import logging
import re

from citerag.chunker import _sentence_spans
from citerag.generator import Citation, Segment, make_citation
from citerag.retriever import tokenize
from citerag.store import Hit

log = logging.getLogger(__name__)

# [3]  [1, 2]  [1-3]  [1–3]
_MARKER_RE = re.compile(r"\[(\d{1,3}(?:\s*[,\-–]\s*\d{1,3})*)\]")
_PARTIAL_RE = re.compile(r"\[[\d,\-–\s]*$")  # a marker that may still be streaming in
_RANGE_RE = re.compile(r"(\d+)\s*[\-–]\s*(\d+)")
# "...Examples 7.8." - the next item's section number, left by the sentence split.
_TRAILING_NUMBER_RE = re.compile(r"\s+\d+(?:\.\d+)+\.?$")
NEAR_BEST = 0.8  # a quote may cover a little less of the claim if it's much shorter
LONG_SENTENCE = 250  # chars; longer "sentences" (tables, lists) are also split into lines


class InlineCitations:
    def __init__(self, hits: list[Hit]):
        self.hits = hits
        self.segments: list[Segment] = []
        self.citations: list[Citation] = []
        self.invalid: list[int] = []  # marker numbers with no matching excerpt
        self._numbers: dict[tuple, int] = {}
        self._pending = ""  # received but not yet processed (may end in a partial marker)
        self._text = ""  # uncited text since the last marker group
        self._claim: Segment | None = None  # cited segment still open for more markers
        self._out: list[str] = []

    @property
    def text(self) -> str:
        """The answer with renumbered markers, as shown to the user."""
        return "".join(self._out).strip()

    def feed(self, piece: str) -> str:
        """Add streamed text; return the display text that is safe to show now."""
        self._pending += piece
        shown = []
        while (m := _MARKER_RE.search(self._pending)) is not None:
            shown.append(self._plain(self._pending[: m.start()]))
            shown.append(self._marker(m.group(1)))
            self._pending = self._pending[m.end():]
        partial = _PARTIAL_RE.search(self._pending)
        cut = partial.start() if partial else len(self._pending)
        shown.append(self._plain(self._pending[:cut]))
        self._pending = self._pending[cut:]
        out = "".join(shown)
        self._out.append(out)
        return out

    def close(self) -> str:
        """End of stream: flush leftover text (an unfinished "[2" is just text)."""
        out = self._plain(self._pending)
        self._pending = ""
        self._out.append(out)
        self._finish_claim()
        if self._text:
            self.segments.append(Segment(self._text))
            self._text = ""
        return out

    # -- internals --

    def _plain(self, text: str) -> str:
        if text:
            if self._claim is not None and text.strip():
                self._finish_claim()  # new words: the previous claim is complete
            self._text += text
        return text

    def _marker(self, group: str) -> str:
        if self._claim is None:
            uncited, claim = _split_last_sentence(self._text)
            if uncited.strip():
                self.segments.append(Segment(uncited))
            self._claim = Segment(claim)
            self._text = ""
        shown = []
        for n in _numbers(group):
            if not 1 <= n <= len(self.hits):
                self.invalid.append(n)
                log.warning("dropped citation [%d]: only %d excerpts were given", n, len(self.hits))
                continue
            number = self._cite(n - 1, self._claim.text)
            if number not in self._claim.citations:
                self._claim.citations.append(number)
                shown.append(f"[{number}]")
        return "".join(shown)

    def _finish_claim(self) -> None:
        if self._claim is not None:
            self.segments.append(self._claim)
            self._claim = None

    def _cite(self, hit_index: int, claim: str) -> int:
        hit = self.hits[hit_index]
        start, end, support = locate_quote(claim, hit.text)
        key = (hit_index, start, end)
        if key not in self._numbers:
            self._numbers[key] = len(self.citations) + 1
            self.citations.append(
                make_citation(self._numbers[key], hit, start, end, hit.text[start:end], support)
            )
        return self._numbers[key]


def locate_quote(claim: str, text: str) -> tuple[int, int, float]:
    """The shortest span of `text` that covers the claim about as well as any.

    Candidates are sentences and pairs of adjacent sentences, plus single and
    paired lines inside very long "sentences" (tables and lists extracted from
    PDFs often have no full stops). Among candidates sharing at least
    NEAR_BEST of the best word overlap, the shortest wins: precise quotes,
    without letting a whole table win just because it contains every word.

    Returns (start, end, support) where support is the share of the claim's
    content words found in the span. With no overlap at all the whole text
    is returned with support 0.0, so the weak citation stays visible.
    """
    claim_words = set(tokenize(claim))
    candidates = _quote_candidates(text)
    if not claim_words or not candidates:
        return 0, len(text), 0.0
    overlap = {span: len(claim_words & set(tokenize(text[span[0]:span[1]]))) for span in candidates}
    best = max(overlap.values())
    if best == 0:
        return 0, len(text), 0.0
    near = [span for span, n in overlap.items() if n >= NEAR_BEST * best]
    start, end = min(near, key=lambda s: (s[1] - s[0], s[0]))
    support = overlap[(start, end)] / len(claim_words)
    trailing = _TRAILING_NUMBER_RE.search(text, start, end)
    if trailing and trailing.start() > start:
        end = trailing.start()
    return start, end, support


def _quote_candidates(text: str) -> set[tuple[int, int]]:
    sentences = _sentence_spans(text, 0, len(text))
    spans = set(sentences)
    spans.update((a[0], b[1]) for a, b in zip(sentences, sentences[1:]))
    for s, e in sentences:
        if e - s > LONG_SENTENCE:
            lines = _line_spans(text, s, e)
            spans.update(lines)
            spans.update((a[0], b[1]) for a, b in zip(lines, lines[1:]))
    return spans


def _line_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    out, pos = [], start
    for line in text[start:end].split("\n"):
        s, e = pos, pos + len(line)
        pos = e + 1
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if any(c.isalnum() for c in text[s:e]):
            out.append((s, e))
    return out


def _numbers(group: str) -> list[int]:
    out = []
    for part in re.split(r"\s*,\s*", group):
        r = _RANGE_RE.fullmatch(part)
        if r:
            a, b = int(r.group(1)), int(r.group(2))
            out.extend(range(a, b + 1) if 0 < b - a < 10 else [a, b])
        elif part.strip().isdigit():
            out.append(int(part))
    return out


def _split_last_sentence(text: str) -> tuple[str, str]:
    """Split off the last sentence (or list item): it's what a marker cites."""
    spans = _sentence_spans(text, 0, len(text))
    if len(spans) <= 1:
        return "", text
    cut = spans[-1][0]
    return text[:cut], text[cut:]
