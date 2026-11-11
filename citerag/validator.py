"""Check that each statement in an answer is backed by what it cites. (Phase 6)

A citation only shows that the model pointed at a passage, not that the
passage says what the sentence claims. After the answer is written, each
sentence or list item is checked:

1. Uncited claims: a statement with no [n] marker. Lead-ins ("The marks
   are:"), headings and "the documents don't say" replies need no citation
   and are skipped.
2. Numbers and names: each number and proper name in the statement must
   appear in a passage it cites. Numbers also match number words and Roman
   numerals ("80" ~ "Eighty", "7" ~ "VII").
3. Support: a small local NLI (natural language inference) model reads the
   cited quote and the cited passage (each after the question, which says
   what the passage is about) and estimates the probability that one of
   them entails the statement, minus any "Based on the documents," lead-in.
   Free and on-device.
4. Invalid markers: [n] markers pointing at excerpts that don't exist
   (Gemini only; Phase 5 drops them from the text) are reported.
5. Tables (2026-11-11): when every number in a statement comes from table
   rows, the table is read instead of asking NLI, which can't tell one
   column of a row from another. Each number must sit in a column and a row
   the statement names ("6,000 is the Daily allowance (NPR) for Grade A
   (Director) in the cited table" otherwise). Form rows also get the lines
   above them as a premise, where their labels and column headings sit.

Each statement gets "ok" (✅) or "warn" (⚠️) with reasons, and the answer a
summary line. Strict mode removes the warned statements and the citations
only they used.

Try it:  python -m citerag.generator "What are the pass marks?" [--strict]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace

import numpy as np

from citerag import config
from citerag.chunker import _sentence_spans
from citerag.generator import DECLINED, NOT_FOUND, Answer, Segment, location
from citerag.retriever import tokenize
from citerag.store import Hit, _best_device

log = logging.getLogger(__name__)

STRICT_EMPTY = "None of the answer's statements could be verified against your documents."
# Below this, an unsupported statement is reported as "doesn't clearly support"
# rather than "contradicts": correct statements wrongly flagged on the syllabus
# set scored 0.06-0.93, most wrong ones 0.83+ (ai_agents/06-citation-checks.md).
CONTRADICTION = 0.8

_BULLET_RE = re.compile(r"^\s*(?:[-*•▪◦]|\d{1,3}[.)])\s+")
_MARKDOWN_RE = re.compile(r"\*\*|__|`|^#+\s*|(?<!\w)\*(?=\S)|(?<=\S)\*(?!\w)")
_HEADING_RE = re.compile(r"[ \t]*(?:#+\s|\*\*[^*\n]+\*\*:?[ \t]*$|__[^_\n]+__:?[ \t]*$)")
# "The documents don't mention ...", "is not specified", "I couldn't find ..."
_DOCS = r"(?:documents?|excerpts?|sources?|passages?|syllabus|text|context|materials?|provided)"
_META_RES = [
    re.compile(r"\b(?:do|does|did)(?:\s+not|n't)\s+(?:\w+ly\s+)?"
               r"(?:mention|say|specify|state|describe|detail|indicate|discuss|address)\b", re.I),
    re.compile(r"\b(?:(?:is|are|was|were)\s+not|isn't|aren't|wasn't|weren't)\s+(?:\w+ly\s+)?"
               r"(?:mentioned|specified|stated|described|detailed|indicated|discussed|addressed)\b", re.I),
    re.compile(r"\bno\s+(?:specific\s+|further\s+|other\s+)?(?:information|mention|details?)\b", re.I),
    re.compile(r"\b(?:couldn't|could not|can't|cannot|unable to)\s+(?:find|locate|determine|answer)\b", re.I),
    re.compile(rf"\b(?:do|does|did)(?:\s+not|n't)\s+(?:\w+ly\s+)?"
               rf"(?:include|contain|provide|cover|list|give|name|identify)\b.*\b{_DOCS}\b"
               rf"|\b{_DOCS}\b.*\b(?:do|does|did)(?:\s+not|n't)\s+(?:\w+ly\s+)?"
               rf"(?:include|contain|provide|cover|list|give|name|identify)\b", re.I),
]

# "Based on the provided documents, ..." adds nothing to check, but it drops the NLI
# score of a correct statement from ~0.99 to ~0.02 (ai_agents/08-evaluation.md).
_LEAD_IN_RE = re.compile(
    rf"^(?:(?:based on|according to)\s+(?:the\s+)?(?:provided\s+|given\s+|cited\s+)?{_DOCS}"
    rf"|(?:in|from)\s+the\s+(?:provided|given|cited)\s+{_DOCS})\b[^,:]{{0,30}}[,:]\s*",
    re.I,
)
_NEGATION_RE = re.compile(r"\b(?:no|not|never|none|nor|without|cannot)\b|n't", re.I)
# LaTeX the LLM writes ($\beta_1 = 0.9$) vs. what the PDF text says (β1 = 0.9).
_MATH_RE = re.compile(r"\$([^$\n]+)\$")
_LATEX = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ", "varepsilon": "ε",
    "theta": "θ", "lambda": "λ", "mu": "μ", "sigma": "σ", "pi": "π", "cdot": "·", "times": "×",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "approx": "≈", "infty": "∞", "pm": "±",
    "sqrt": "√", "to": "→", "ldots": "...", "dots": "...",
}
_DISPLAY_MARKER_RE = re.compile(r"\[(\d+)\]")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_ROMAN_RE = re.compile(r"\b[IVXL]{1,6}\b")
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}
_NUMBER_WORDS = {
    w: i for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
} | {"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
     "ninety": 90, "hundred": 100}
_ORDINALS = {
    w: i for i, w in enumerate(
        "first second third fourth fifth sixth seventh eighth ninth tenth".split(), 1
    )
}
_WORD_RE = re.compile(r"[A-Za-z][\w.+#&'-]*")


@dataclass
class Check:
    text: str  # the statement, without markdown or [n] markers
    citations: list[int]  # Citation.number values
    status: str  # "ok" (cited and supported), "warn", or "skip" (nothing to check)
    issues: list[str] = field(default_factory=list)
    entailment: float | None = None  # best NLI score against what it cites (0..1)
    start: int = 0  # span in the answer's plain text (its segments joined)
    end: int = 0


@dataclass
class Validation:
    checks: list[Check]
    invalid_markers: list[int] = field(default_factory=list)
    removed: list[Check] = field(default_factory=list)  # by strict mode

    @property
    def claims(self) -> list[Check]:
        return [c for c in self.checks if c.status != "skip"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == "warn"]

    @property
    def ok(self) -> bool:
        return not self.warnings and not self.invalid_markers

    def summary(self) -> str:
        claims, warned = self.claims, len(self.warnings)
        if claims:
            text = f"{len(claims) - warned}/{len(claims)} statements supported by their citations"
            if warned:
                text += f", {warned} warning{'s' if warned > 1 else ''}"
        elif self.removed:
            text = "No statement could be verified"
        else:
            text = "No statements to check"
        if self.removed:
            n = len(self.removed)
            text += f"; strict mode removed {n} unsupported statement{'s' if n > 1 else ''}"
        if self.invalid_markers:
            n = len(self.invalid_markers)
            markers = ", ".join(f"[{m}]" for m in self.invalid_markers)
            text += f"; {n} invalid citation marker{'s' if n > 1 else ''} dropped ({markers})"
        return text


class NLI:
    """Cross-encoder NLI model: (premise, hypothesis) -> P(entailment), P(contradiction)."""

    def __init__(
        self,
        model_name: str = config.NLI_MODEL,
        device: str | None = config.EMBED_DEVICE,
        batch_size: int = config.EMBED_BATCH_SIZE,
        max_tokens: int = 512,
    ):
        self.model_name = model_name
        self.device = device or _best_device()
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self._model = None
        self._labels: dict[str, int] = {}

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading NLI model %s on %s", self.model_name, self.device)
            self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_tokens)
            labels = {v.lower(): int(k) for k, v in self._model.config.id2label.items()}
            self._labels = {
                "entailment": next(i for name, i in labels.items() if name.startswith("entail")),
                "contradiction": next((i for name, i in labels.items() if name.startswith("contra")), -1),
            }
        return self._model

    def scores(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """(premise, hypothesis) pairs -> array of shape (n, 2): entailment, contradiction."""
        if not pairs:
            return np.zeros((0, 2), dtype=np.float32)
        probs = self.model.predict(
            pairs, batch_size=self.batch_size, apply_softmax=True, show_progress_bar=False
        )
        probs = np.asarray(probs, dtype=np.float32).reshape(len(pairs), -1)
        entail, contra = self._labels["entailment"], self._labels["contradiction"]
        contradiction = probs[:, contra] if contra >= 0 else np.zeros(len(pairs), dtype=np.float32)
        return np.stack([probs[:, entail], contradiction], axis=1)


class Validator:
    def __init__(self, nli: NLI | None = None, threshold: float = config.NLI_THRESHOLD):
        self.nli = nli or NLI()
        self.threshold = threshold

    def validate(self, answer: Answer) -> Validation:
        cited = {c.number: c for c in answer.citations}
        hits = {h.chunk_id: h for h in answer.hits}
        plain = _plain_text(answer.segments)
        checks, pending = [], []  # pending: (check, premises, cited hits) awaiting NLI
        for start, end, nums in statement_spans(answer.segments):
            raw = plain[start:end]
            text = clean(raw)
            nums = [n for n in nums if n in cited]
            check = Check(text, nums, "ok", start=start, end=end)
            checks.append(check)
            # Whatever the model chose to cite is checked, even a lead-in like
            # "Unit VII covers these topics [1]:".
            if _HEADING_RE.match(raw) or not (nums or needs_citation(text)):
                check.status = "skip"
                continue
            if not nums:
                pending.append((check, [h.text for h in answer.hits], answer.hits))
                continue
            sources = list({cited[n].chunk_id: hits[cited[n].chunk_id] for n in nums
                            if cited[n].chunk_id in hits}.values())
            missing = missing_terms(text, sources)
            if missing:
                # Answers restate the question's subject ("the pass marks for the Web Technology
                # exam are 45") that the cited passage only implies. A term the question uses
                # only has to occur in some retrieved passage, so a false premise is still caught.
                retrieved = set(missing_terms(text, answer.hits))
                asked = set(tokenize(answer.question))
                missing = [m for m in missing
                           if m in retrieved or not (tokenize(m) and set(tokenize(m)) <= asked)]
            if missing:
                check.issues.append(f"not in {_cited(sources)}: " + ", ".join(f'"{m}"' for m in missing))
            quotes = [cited[n].cited_text for n in nums]
            premises = quotes + ([" ".join(quotes)] if len(quotes) > 1 else []) + [h.text for h in sources]
            claim = _claim(text)
            if _is_term(claim):
                # A bare list item ("Selectors") isn't a sentence an NLI model can judge;
                # what matters is that what it cites names it.
                if not set(tokenize(claim)) <= set(tokenize(" ".join(premises))):
                    check.issues.append(f"{_cited(sources)} {'doesn' if len(sources) == 1 else 'don'}'t mention it")
                check.status = "warn" if check.issues else "ok"
                continue
            if (verdict := table_verdict(claim, sources)) is not None:
                # Every number is a table cell: read the table instead of asking NLI, which can't
                # tell one column of a row from another (any value in the row "fits" the claim).
                check.issues.extend(verdict)
                check.status = "warn" if check.issues else "ok"
                continue
            premises += line_windows(claim, sources)
            pending.append((check, list(dict.fromkeys(premises)), sources))

        # The question goes in with each passage: it names what the passage is about, which
        # the passage itself often doesn't ("The encoder is ..." in the Transformer paper).
        pairs = [(f"{answer.question}\n{p}", _claim(check.text).rstrip(":"))
                 for check, premises, _ in pending for p in premises]
        scores = self.nli.scores(pairs) if pairs else np.zeros((0, 2))
        i = 0
        for check, premises, sources in pending:
            s = scores[i:i + len(premises)]
            i += len(premises)
            best = int(s[:, 0].argmax()) if len(s) else -1
            check.entailment = round(float(s[best, 0]), 3) if len(s) else None
            supported = check.entailment is not None and check.entailment >= self.threshold
            if not check.citations:
                if supported:
                    h = sources[best]
                    check.issues.append(f"no citation, though {location(h.source, h.page_start, h.page_end)} supports it")
                else:
                    check.issues.append("no citation, and no retrieved passage supports it")
            elif not supported:
                one = len(sources) == 1
                if len(s) and float(s[:, 1].max()) >= CONTRADICTION:
                    check.issues.append(f"{_cited(sources)} {'contradicts' if one else 'contradict'} it")
                else:
                    check.issues.append(f"{_cited(sources)} {'doesn' if one else 'don'}'t clearly support it")
            check.status = "warn" if check.issues else "ok"
        return Validation(checks, list(answer.invalid_markers))


def strict(answer: Answer) -> Answer:
    """A copy of the answer without its warned statements, renumbering the citations left."""
    v = answer.validation
    if v is None or not v.warnings:
        return answer
    removed = [c for c in v.checks if c.status == "warn"]
    plain = _plain_text(answer.segments)
    cuts = []  # (start, end) to delete, each with the whitespace that follows it
    for c in removed:
        end = c.end
        while end < len(plain) and plain[end].isspace():
            end += 1
        cuts.append((c.start, end))

    segments, pos = [], 0
    for seg in answer.segments:
        a, b = pos, pos + len(seg.text)
        pos = b
        kept = "".join(ch for i, ch in enumerate(seg.text, a) if not _inside(i, cuts))
        if not kept:
            continue
        last = a + len(seg.text.rstrip()) - 1  # its [n] markers follow this character
        cites = seg.citations if seg.citations and not _inside(last, cuts) else []
        segments.append(Segment(kept, list(cites)))

    order = list(dict.fromkeys(n for s in segments for n in s.citations))
    renumber = {old: new for new, old in enumerate(order, 1)}
    citations = [replace(c, number=renumber[c.number]) for c in answer.citations if c.number in renumber]
    citations.sort(key=lambda c: c.number)
    for s in segments:
        s.citations = [renumber[n] for n in s.citations]
    kept_checks = []
    for c in v.checks:
        if c.status == "warn":
            continue
        shift = sum(e - s for s, e in cuts if e <= c.start)
        kept_checks.append(replace(c, citations=[renumber[n] for n in c.citations if n in renumber],
                                   start=c.start - shift, end=c.end - shift))
    validation = Validation(kept_checks, v.invalid_markers, removed)

    if not validation.claims:
        return replace(answer, text=STRICT_EMPTY, segments=[Segment(STRICT_EMPTY)], citations=[],
                       found=False, validation=Validation([], v.invalid_markers, removed))
    text = _cut_text(answer.text, plain, cuts, renumber)
    return replace(answer, text=text, segments=segments, citations=citations,
                   found=bool(citations), validation=validation)


def display_offsets(text: str, plain: str) -> list[int]:
    """Where each character of `plain` (and its end) sits in `text`.

    `text` is the answer as displayed: `plain` (the segments joined) with [n]
    markers inserted and outer whitespace stripped. Each backend places
    markers its own way, so the two are aligned rather than re-rendered. A
    marker goes with the character before it: the span [pos[a], pos[b]) of a
    statement includes the markers right after it.
    """
    pos, i = [], 0
    for j, ch in enumerate(plain):
        while (m := _DISPLAY_MARKER_RE.match(text, i)) and not plain.startswith(m.group(), j):
            i = m.end()
        pos.append(i)
        if i < len(text) and text[i] == ch:
            i += 1  # otherwise ch was whitespace stripped from the display text
    while m := _DISPLAY_MARKER_RE.match(text, i):
        i = m.end()
    pos.append(i)
    return pos


def _cut_text(text: str, plain: str, cuts: list[tuple[int, int]], renumber: dict[int, int]) -> str:
    """Delete the cut spans of `plain` (with their markers) from the displayed text."""
    pos = display_offsets(text, plain)
    removed = [False] * len(text)
    for s, e in cuts:
        removed[pos[s]:pos[e]] = [True] * (pos[e] - pos[s])
    plain_at = set(pos[:-1])
    out, i = [], 0
    while i < len(text):
        m = _DISPLAY_MARKER_RE.match(text, i)
        if m and i not in plain_at:
            n = int(m.group(1))
            if not removed[i] and n in renumber:
                out.append(f"[{renumber[n]}]")
            i = m.end()
            continue
        if not removed[i]:
            out.append(text[i])
        i += 1
    return "".join(out).strip()


# -- statements ----------------------------------------------------------------------


def statement_spans(segments: list[Segment]) -> list[tuple[int, int, list[int]]]:
    """Split the answer into sentences and lines: (start, end, citation numbers).

    Offsets are into the plain text (segment texts joined, no markers). A
    statement carries the citations of every cited segment it overlaps.
    """
    plain = _plain_text(segments)
    bounds, pos = [], 0
    for s in segments:
        bounds.append((pos, pos + len(s.text), s.citations))
        pos += len(s.text)
    out = []
    for s, e in _line_spans(plain):
        heading = _HEADING_RE.match(plain, s, e) is not None  # "**1. Marks**" is one unit
        for a, b in [(s, e)] if heading else _sentence_spans(plain, s, e):
            if not any(ch.isalnum() for ch in plain[a:b]):
                continue
            nums: list[int] = []
            for x, y, cites in bounds:
                if cites and x < b and a < y and plain[max(x, a):min(y, b)].strip():
                    nums += [n for n in cites if n not in nums]
            out.append((a, b, nums))
    return out


def clean(text: str) -> str:
    """Drop list bullets and markdown emphasis, turn $LaTeX$ into plain text; collapse whitespace."""
    text = _MARKDOWN_RE.sub("", _BULLET_RE.sub("", text, count=1))
    text = _MATH_RE.sub(lambda m: plain_math(m.group(1)), text)
    text = " ".join(text.split())
    return re.sub(r"\s+([.,;:!?)])", r"\1", text)


def plain_math(tex: str) -> str:
    r"""LaTeX math as plain text, the way PDF text shows it: "\beta_1 = 0.9" -> "β1 = 0.9"."""
    tex = re.sub(r"\\(?:mathrm|text|mathbf|textbf|mathit|operatorname)\{([^}]*)\}", r"\1", tex)
    tex = re.sub(r"\\sqrt\{([^{}]*)\}", r"√\1", tex)
    tex = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", tex)  # \frac{1}{\sqrt{d_k}} -> 1/√d_k
    tex = re.sub(r"\\([A-Za-z]+)", lambda m: _LATEX.get(m.group(1), m.group(1)), tex)
    return re.sub(r"[_^{}]|\\[,;! ]", "", tex)


def _is_term(text: str) -> bool:
    """A bare term or short list item ("Selectors", "DOM tree") rather than a sentence.

    Numbers and negations are left to the NLI model: "Pass marks: 100" could pair a
    label with the wrong value from a table, and "Not mandatory" says the opposite.
    """
    words = tokenize(text)
    return 0 < len(words) <= 3 and not re.search(r"\d", text) and not _NEGATION_RE.search(text)


def _claim(text: str) -> str:
    """The part of a statement that makes a claim, without a "Based on the documents," lead-in.

    A leading "Yes," goes too: what follows says the same. "No," stays, it carries the claim.
    """
    return re.sub(r"^yes[,.!]\s+", "", _LEAD_IN_RE.sub("", text, count=1), count=1, flags=re.I)


def needs_citation(text: str) -> bool:
    """False for lead-ins, "not in the documents" statements and fragments."""
    if text.rstrip().endswith(":") or text.strip() in (NOT_FOUND, DECLINED):
        return False
    if any(r.search(text) for r in _META_RES):
        return False
    return len(tokenize(text)) >= 2


def missing_terms(text: str, sources: list[Hit]) -> list[str]:
    """Numbers and names in the statement that none of the sources contains."""
    source_text = " ".join(f"{h.source} {h.section or ''} {h.text}" for h in sources)
    numbers = _source_numbers(source_text) | {str(p) for h in sources for p in (h.page_start, h.page_end)}
    words = set(tokenize(source_text))
    initials = "".join(w[0] for w in re.findall(r"[A-Za-z]+", source_text)).lower()
    missing = []
    for n in _NUMBER_RE.findall(text):
        if _normalize_number(n) not in numbers and n not in missing:
            missing.append(n)
    for name in _names(text):
        if name.isupper() and len(name) >= 3 and name.lower() in initials:
            continue  # "CSS" for "Cascading Style Sheets"
        if any(len(t) > 1 and t not in words and not t.isdigit() for t in tokenize(name)):
            if name not in missing:
                missing.append(name)
    return missing


_TABLE_ROW_RE = re.compile(r"^- (?=.*: )(?=.*; ).+$")  # "- Grade: A; Daily allowance (NPR): 6,000; …"


def _line_numbers(line: str) -> set[str]:
    return {_normalize_number(n) for n in _NUMBER_RE.findall(line)}


# A number on its own, not part of a name or a date ("ViT-G/14", "ImageNet-v2", "01/02/2022", "±0.03").
_STANDALONE_RE = re.compile(r"(?<![\w/.^±-])\d+(?:[.,]\d+)*(?![\w/^-]|\.\d)")


def _numbers_in(text: str) -> set[str]:
    return {_normalize_number(n) for n in _STANDALONE_RE.findall(text)}


def table_verdict(claim: str, sources: list[Hit]) -> list[str] | None:
    """Check a claim whose numbers all come from table rows by reading the table.

    Tables are written one row per line ("- Grade: A (Director); Daily
    allowance (NPR): 6,000; Hotel limit per night (NPR) Kathmandu: 18,000",
    Phase 9). Each number in the claim must sit in a column and a row the claim
    names. A column or row is named by the words that tell it apart from the
    other columns of its row or the other rows of its table ("Kathmandu" vs
    "Other cities", "Director" vs "Manager"); a header word may be a prefix
    of the claim's word ("FR" ~ "French", "params" ~ "parameters").

    Returns [] when every number is in a column and row the claim names
    (supported), a reason per misplaced number when the claim names another
    column or row instead ("6,000 is the Daily allowance (NPR) for Grade: A
    (Director) in the cited table"), or None to leave the claim to NLI: a
    number also in the prose, or a table whose columns or rows can't be
    told apart by words.
    """
    numbers = _numbers_in(claim)
    if not numbers:
        return None
    words = set(tokenize(claim))

    def named(tokens: set[str]) -> bool:
        return any(t in words or (len(t) >= 2 and any(w.startswith(t) for w in words)) for t in tokens)

    def distinct(own: str, others: list[str]) -> set[str]:
        rest = {t for o in others for t in tokenize(o)}
        return {t for t in tokenize(own) if t not in rest and not t.isdigit()}

    found: dict[str, list[tuple[list, int, int]]] = {n: [] for n in numbers}  # number -> (block, row, cell)
    for h in sources:
        block: list[list[tuple[str, str]]] = []
        for line in h.text.split("\n") + [""]:
            line = line.strip()
            if _TABLE_ROW_RE.match(line):
                cells = [c.rpartition(": ") if ": " in c else ("", "", c) for c in line[2:].split("; ")]
                block.append([(head, value) for head, _, value in cells])
                continue
            if numbers & _numbers_in(line):
                return None  # also in the prose: not a table claim
            for r, row in enumerate(block):
                for c, (_, value) in enumerate(row):
                    for n in numbers & _numbers_in(value):
                        found[n].append((block, r, c))
            block = []
    if not all(found.values()):
        return None

    reasons = []
    for n, places in found.items():
        verdicts = []
        for block, r, c in places:
            row = block[r]
            label = " ".join(filter(None, row[0]))  # the first cell names the row
            numeric = [i for i, (_, v) in enumerate(row) if _numbers_in(v) and i != c]
            col_words = distinct(row[c][0], [row[i][0] for i in numeric])
            other_cols = [distinct(row[i][0], [row[j][0] for j in numeric + [c] if j != i]) for i in numeric]
            labels = [" ".join(filter(None, other[0])) for other in block]
            row_words = distinct(label, [l for i, l in enumerate(labels) if i != r]) if len(block) > 1 else None
            other_rows = [distinct(l, [x for j, x in enumerate(labels) if j != i]) for i, l in enumerate(labels) if i != r]
            # Named: the claim uses the number's own column (row) and no other; wrong: another and not its own.
            own_col, any_col = bool(col_words) and named(col_words), any(o and named(o) for o in other_cols)
            col_ok, col_wrong = own_col and not any_col, any_col and not own_col
            if row_words is None or c == 0:  # one row, or the number is the row's own label
                row_ok, row_wrong = True, False
            else:
                own_row, any_row = bool(row_words) and named(row_words), any(o and named(o) for o in other_rows)
                row_ok, row_wrong = own_row and not any_row, any_row and not own_row
            if col_ok and row_ok:
                verdicts.append("ok")
            elif col_wrong or row_wrong:
                head, value = row[c][0] or "value", row[c][1]
                verdicts.append(f"{value} is the {head} for {label} in the cited table")
            else:
                verdicts.append(None)
        if "ok" in verdicts:
            continue
        if None in verdicts:
            return None  # can't tell: leave the claim to NLI
        reasons.append(verdicts[0])
    return reasons


def line_windows(claim: str, sources: list[Hit], above: int = 2) -> list[str]:
    """Each form-like line holding a number from the claim, with the lines above it.

    On a form (Phase 10, read row by row) a value's label or column heading is
    often a row or two above it ("2b Employer Identification Number" ...
    "735268329"). Form rows separate fields by three spaces.
    """
    numbers = {_normalize_number(n) for n in _NUMBER_RE.findall(claim)}
    out = []
    for h in sources:
        lines = [line for line in h.text.split("\n") if line.strip()]
        for i, line in enumerate(lines):
            if "   " in line and numbers & _line_numbers(line):
                out.append("\n".join(lines[max(0, i - above):i + 1]))
    return out


def _names(text: str) -> list[str]:
    """Capitalised words, acronyms and mixed-case terms, except where any word
    could be capitalised: first in the statement or after a colon/semicolon."""
    out = []
    for i, m in enumerate(_WORD_RE.finditer(text)):
        if i == 0 or re.search(r"[:;]\s*$", text[:m.start()]):
            continue
        w = m.group().rstrip(".'-")
        if w[:1].isupper() or any(c.isupper() for c in w[1:]) or any(c.isdigit() for c in w):
            out.append(w)
    return out


def _source_numbers(text: str) -> set[str]:
    nums = set()
    for n in _NUMBER_RE.findall(text):
        nums.add(_normalize_number(n))
        nums.update(re.split(r"[.,]", n))  # "7.7" -> also "7"; "60,20" -> "60", "20"
    for n in re.findall(r"\b\d{1,3}(?:[ \u2009\u202f]\d{3})+\b", text):
        nums.add(re.sub(r"\D", "", n))  # "10 000" (thousands separated by a space) -> "10000"
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in _NUMBER_WORDS:
            nums.add(str(_NUMBER_WORDS[w]))
        elif w in _ORDINALS:
            nums.add(str(_ORDINALS[w]))
    for r in _ROMAN_RE.findall(text):
        nums.add(str(_roman(r)))
    return nums


def _normalize_number(n: str) -> str:
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", n):
        return n.replace(",", "")  # 1,000 -> 1000
    return n.rstrip(".,")


def _roman(s: str) -> int:
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = _ROMAN[a]
        total += -v if b != " " and _ROMAN[b] > v else v
    return total


def _line_spans(text: str) -> list[tuple[int, int]]:
    """LLM answers aren't hard-wrapped: each line is its own statement or item."""
    out, pos = [], 0
    for line in text.split("\n"):
        if line.strip():
            out.append((pos, pos + len(line)))
        pos += len(line) + 1
    return out


def _cited(sources: list[Hit]) -> str:
    return "the cited passage" if len(sources) == 1 else "the cited passages"


def _plain_text(segments: list[Segment]) -> str:
    return "".join(s.text for s in segments)


def _inside(i: int, cuts: list[tuple[int, int]]) -> bool:
    return any(s <= i < e for s, e in cuts)
