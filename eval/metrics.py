"""Scoring for the Phase 8 evaluation: did we find the passage, and did the answer get it right?

Gold labels are exact quotes, not chunk IDs, so the same question set works for
any chunk size: each quote is located once in the document's text (the text the
chunker splits), and a retrieved chunk "covers" a piece of evidence when its
span contains one of that piece's quotes whole. A fact a paper repeats (abstract,
results, tables) lists every place it appears, so finding any of them counts.

Answers are scored on plain records (see `eval.answers.record`), so cached
answers can be re-scored without calling the LLM again.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from citerag.validator import _META_RES

KINDS = ("single", "multi", "unanswerable", "off_topic")
ANSWERABLE = ("single", "multi")

# Typographic variants the PDF text and a hand-typed quote may disagree on.
_CHARS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-"})
_MARKER_RE = re.compile(r"\[\d+\]")


@dataclass
class Evidence:
    """One piece of the answer. It may appear in several places: any of `quotes` will do."""

    source: str
    quotes: list[str]
    spans: list[tuple[int, int]] = field(default_factory=list)  # in the document's joined text


@dataclass
class Question:
    id: str
    question: str
    kind: str
    evidence: list[Evidence]
    facts: list[list[str]]  # every group must match; any alternative in a group will do
    forbidden: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return self.kind in ANSWERABLE

    @property
    def sources(self) -> set[str]:
        return {e.source for e in self.evidence}


def load_questions(path: str | Path) -> tuple[list[str], list[Question]]:
    """(document file names, questions) from a question-set JSON file."""
    data = json.loads(Path(path).read_text())
    questions = [
        Question(
            id=q["id"],
            question=q["question"],
            kind=q["kind"],
            evidence=[Evidence(e["source"], [e["quote"]] if isinstance(e["quote"], str) else e["quote"])
                      for e in q.get("evidence", [])],
            facts=q.get("facts", []),
            forbidden=q.get("forbidden", []),
        )
        for q in data["questions"]
    ]
    return data["documents"], questions


def problems(documents: list[str], questions: list[Question]) -> list[str]:
    """Mistakes in the question set itself (before any document is read)."""
    out, seen = [], set()
    for q in questions:
        if q.id in seen:
            out.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if q.kind not in KINDS:
            out.append(f"{q.id}: unknown kind {q.kind!r}")
        if q.answerable and not (q.evidence and q.facts):
            out.append(f"{q.id}: answerable questions need evidence and facts")
        if not q.answerable and q.evidence:
            out.append(f"{q.id}: {q.kind} questions can't have evidence")
        for e in q.evidence:
            if e.source not in documents:
                out.append(f"{q.id}: {e.source} is not in the document list")
    return out


# -- locating quotes ----------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase, unify quotes and dashes, collapse whitespace."""
    return " ".join(text.translate(_CHARS).lower().split())


def find_span(text: str, quote: str) -> tuple[int, int] | None:
    """Span of `quote` in `text`, ignoring case, typographic quotes and whitespace."""
    target = normalize(quote)
    norm, index = [], []  # index[i] = position in `text` of norm[i]
    space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            space = bool(norm)
            continue
        if space:
            norm.append(" ")
            index.append(i - 1)
            space = False
        for c in ch.translate(_CHARS).lower():  # lower() can return two characters
            norm.append(c)
            index.append(i)
    at = "".join(norm).find(target) if target else -1
    if at < 0:
        return None
    return index[at], index[at + len(target) - 1] + 1


def locate(questions: list[Question], texts: dict[str, str]) -> list[str]:
    """Set every evidence span from the documents' texts; returns quotes not found."""
    missing = []
    for q in questions:
        for e in q.evidence:
            e.spans = []
            for quote in e.quotes:
                span = find_span(texts.get(e.source, ""), quote)
                if span is None:
                    missing.append(f"{q.id}: quote not found in {e.source}: {quote!r}")
                else:
                    e.spans.append(span)
    return missing


# -- retrieval ----------------------------------------------------------------------


def covers(source: str, start: int, end: int, e: Evidence) -> bool:
    """True when the span [start, end) of `source` contains one of the quotes whole."""
    return source == e.source and any(start <= s and t <= end for s, t in e.spans)


def retrieval_scores(q: Question, hits: list, k: int) -> dict:
    """How well the top-k hits (objects with source/start_char/end_char) cover the evidence.

    recall: share of the evidence quotes inside the top k; all: every quote is;
    rr: 1 / rank of the first hit holding any quote (0 if none in the top k);
    top_doc: the first hit comes from a document the question is about.
    """
    top = hits[:k]
    found = [any(covers(h.source, h.start_char, h.end_char, e) for h in top) for e in q.evidence]
    rank = next(
        (i for i, h in enumerate(top, 1)
         if any(covers(h.source, h.start_char, h.end_char, e) for e in q.evidence)),
        None,
    )
    return {
        "recall": sum(found) / len(found),
        "all": all(found),
        "rr": 1 / rank if rank else 0.0,
        "top_doc": bool(top) and top[0].source in q.sources,
    }


# -- answers ------------------------------------------------------------------------


def plain(text: str) -> str:
    """Answer text without [n] markers or markdown emphasis, normalized for matching."""
    return normalize(_MARKER_RE.sub("", text).replace("**", "").replace("__", ""))


def has_fact(text: str, alternatives: list[str]) -> bool:
    t = plain(text)
    return any(normalize(a) in t for a in alternatives)


def says_not_found(text: str) -> bool:
    """The answer states that the documents don't contain (part of) the answer."""
    return any(r.search(text) for r in _META_RES)


def answer_scores(q: Question, rec: dict) -> dict:
    """Outcome of one answer record: correct | partial | wrong | refused (+ citation checks).

    For unanswerable and off-topic questions the right outcome is "the documents
    don't say" (whether the gate stopped it or the model said so) without any
    of the forbidden facts.
    """
    text = rec["text"]
    forbidden = [f for f in q.forbidden if has_fact(text, [f])]
    declined = not rec["found"] or says_not_found(text)
    out = {"forbidden": forbidden}
    if not q.answerable:
        out["outcome"] = "correct" if declined and not forbidden else "wrong"
        return out

    matched = sum(has_fact(text, group) for group in q.facts)
    # A question about two documents answered from only one of them is incomplete,
    # even when the facts happen to appear (e.g. one paper mentioning the other's method).
    cites_every_doc = q.kind != "multi" or q.sources <= {c["source"] for c in rec["citations"]}
    if matched == len(q.facts) and not forbidden and cites_every_doc:
        out["outcome"] = "correct"
    elif matched and not forbidden:
        out["outcome"] = "partial"
    elif declined:  # said the documents don't give it, possibly citing related passages
        out["outcome"] = "refused"
    else:
        out["outcome"] = "wrong"
    out["facts"] = f"{matched}/{len(q.facts)}"

    chunks = {h["chunk_id"]: h for h in rec["hits"]}
    cites = rec["citations"]
    out["cited"] = bool(cites)
    # The cited passage holds an evidence quote (the right place to look) ...
    out["cites_evidence"] = any(
        c["chunk_id"] in chunks
        and any(covers(c["source"], chunks[c["chunk_id"]]["start_char"],
                       chunks[c["chunk_id"]]["end_char"], e) for e in q.evidence)
        for c in cites
    )
    # ... and the highlighted quote itself overlaps one (the exact place).
    out["quote_on_evidence"] = any(
        c["source"] == e.source and c["doc_start_char"] < t and s < c["doc_end_char"]
        for c in cites for e in q.evidence for s, t in e.spans
    )
    out["wrong_doc_citations"] = sum(c["source"] not in q.sources for c in cites)
    return out
