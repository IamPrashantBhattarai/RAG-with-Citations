"""Benchmark for the citation checker: false alarms on true statements, catches on altered ones.

Uses saved answer runs (eval/runs/answers-<tag>.jsonl), so no LLM is called.

- True statements: every checked statement (cited, not skipped) in the answers
  scored correct.
- Altered statements: in each true statement, every number that a cited
  passage also contains is replaced, one at a time, by a wrong one:
    neighbour  another number from the same table row or line of the cited
               passage ("Kathmandu: 12,000" for 18,000). This is the hard case:
               the wrong number is right there in the passage, under another label.
    near       one digit changed, when the line has no other number.
  The altered answer is checked again and the altered statement's verdict read back.

A good checker warns on few true statements and on most altered ones.

Run:  python -m eval.checks --tags docling-lite ocr-lite docx-lite
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from dataclasses import replace

from citerag import config
from citerag.generator import Answer, Segment
from citerag.store import Embedder
from citerag.validator import Validator, _normalize_number
from eval.answers import load_records, rebuild
from eval.metrics import load_questions
from eval.retrieval import QUESTIONS, RUNS_DIR, load_documents, open_index

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
# A standalone number: not part of a name or a date ("ViT-G/14", "β1", "01/02/2022", "10^-9").
_STANDALONE_RE = re.compile(r"(?<![\w/.^±-])\d+(?:[.,]\d+)*(?![\w/^-]|\.\d)")


def alterations(text: str, sources: list[str]) -> list[tuple[int, int, str, str]]:
    """(start, end, wrong number, kind) for each number in `text` that a source line contains."""
    out = []
    for m in _STANDALONE_RE.finditer(text):
        n = _normalize_number(m.group())
        lines = [line for s in sources for line in s.split("\n")
                 if n in {_normalize_number(x) for x in _STANDALONE_RE.findall(line)}]
        if not lines:
            continue
        others = [x for line in lines for x in _STANDALONE_RE.findall(line)
                  if _normalize_number(x) != n and n not in _normalize_number(x) and _normalize_number(x) not in n]
        if others:
            out.append((m.start(), m.end(), others[0], "neighbour"))
        else:
            digits = m.group()
            i = max(i for i, c in enumerate(digits) if c.isdigit())
            wrong = digits[:i] + str((int(digits[i]) + 1) % 10) + digits[i + 1:]
            out.append((m.start(), m.end(), wrong, "near"))
    return out


def altered(answer: Answer, start: int, end: int, wrong: str) -> Answer:
    """The answer with plain-text span [start, end) of its segments replaced by `wrong`."""
    segments, pos = [], 0
    for seg in answer.segments:
        s, e = pos, pos + len(seg.text)
        text = seg.text
        if s <= start and end <= e:
            text = text[:start - s] + wrong + text[end - s:]
        segments.append(Segment(text, seg.citations))
        pos = e
    return replace(answer, segments=segments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Citation-checker benchmark on saved answers.")
    parser.add_argument("--tags", nargs="+", required=True, help="saved runs (answers-<tag>.jsonl)")
    parser.add_argument("--show", type=int, default=15, help="false alarms and misses to list")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    documents, _ = load_questions(QUESTIONS)
    pages, _ = load_documents(documents)
    store = open_index(config.CHUNK_SIZE, pages, Embedder())
    validator = Validator()

    true_rows, alt_rows = [], []
    for tag in args.tags:
        records = load_records(RUNS_DIR / f"answers-{tag}.jsonl")
        scores = json.loads((RUNS_DIR / f"scores-{tag}.json").read_text())
        for qid, rec in records.items():
            if not rec["hits"] or scores.get(qid, {}).get("outcome") != "correct":
                continue
            answer = rebuild(rec, store)
            checks = validator.validate(answer).checks
            cited = {c.number: c for c in answer.citations}
            hits = {h.chunk_id: h for h in answer.hits}
            for i, c in enumerate(checks):
                if c.status == "skip" or not c.citations:
                    continue
                true_rows.append({"id": qid, "text": c.text, "warned": c.status == "warn", "issues": c.issues})
                sources = [hits[cited[n].chunk_id].text for n in c.citations if cited[n].chunk_id in hits]
                statement = "".join(s.text for s in answer.segments)[c.start:c.end]
                for start, end, wrong, kind in alterations(statement, sources):
                    a2 = altered(answer, c.start + start, c.start + end, wrong)
                    c2 = validator.validate(a2).checks[i]
                    alt_rows.append({"id": qid, "kind": kind, "text": c2.text, "caught": c2.status == "warn",
                                     "was": statement[start:end], "now": wrong})

    alarms = [r for r in true_rows if r["warned"]]
    print(f"\nTrue statements warned (false alarms): {len(alarms)}/{len(true_rows)} "
          f"({len(alarms) / len(true_rows):.0%})")
    for kind in ("neighbour", "near"):
        rows = [r for r in alt_rows if r["kind"] == kind]
        if rows:
            print(f"Altered statements caught, {kind:<9}: {sum(r['caught'] for r in rows)}/{len(rows)} "
                  f"({statistics.mean(r['caught'] for r in rows):.0%})")
    caught = sum(r["caught"] for r in alt_rows)
    tpr, tnr = caught / len(alt_rows), 1 - len(alarms) / len(true_rows)
    print(f"Altered statements caught, all      : {caught}/{len(alt_rows)} ({tpr:.0%});"
          f" balanced accuracy {(tpr + tnr) / 2:.3f}")
    print("\nFalse alarms:")
    for r in alarms[:args.show]:
        print(f"  {r['id']}: {r['text'][:110]} -> {'; '.join(r['issues'])}")
    print("\nMissed alterations:")
    for r in [r for r in alt_rows if not r["caught"]][:args.show]:
        print(f"  {r['id']} ({r['kind']}, {r['was']} -> {r['now']}): {r['text'][:110]}")


if __name__ == "__main__":
    main()
