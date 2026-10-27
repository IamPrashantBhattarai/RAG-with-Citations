"""Answer-level evaluation: the whole pipeline (retrieval -> LLM -> checks) on the question set.

Each answer is saved as soon as it arrives (eval/runs/answers-<tag>.jsonl), so a
run stopped by a rate limit resumes where it left off, and saved answers can be
re-scored without calling the LLM again (--score-only).

Scored per question (see eval/metrics.py): correct / partial / wrong / refused
from the facts the answer contains; whether a citation points at a passage
holding the evidence, and whether its highlighted quote overlaps the evidence;
citations to the wrong document; and whether the Phase 6 checks warned.

Run:  python -m eval.answers --tag hybrid                          # current settings
      python -m eval.answers --tag rerank --rerank --candidates 10
      python -m eval.answers --tag hybrid --score-only               # re-score saved answers
      python -m eval.answers --tag hybrid --recheck                  # ...after re-running the checks
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from citerag import config
from citerag.citations import _MARKER_RE, _numbers, _split_last_sentence
from citerag.generator import Answer, Citation, GenerationError, Generator, Segment, make_backend
from citerag.retriever import Reranker, Retriever
from citerag.store import Embedder
from citerag.validator import Validator
from eval.metrics import Question, answer_scores, load_questions, locate, problems
from eval.retrieval import QUESTIONS, RUNS_DIR, load_documents, open_index

MAX_FAILURES = 3  # consecutive LLM errors (usually the free-tier quota) before stopping
RETRY_WAIT = 60  # seconds; free-tier per-minute quotas reset within a minute


def record(q: Question, answer: Answer, seconds: float) -> dict:
    """What scoring needs from an Answer, as plain JSON."""
    checks = answer.validation.checks if answer.validation else []
    return {
        "id": q.id,
        "question": q.question,
        "text": answer.text,
        "found": answer.found,
        "model": answer.model,
        "stop_reason": answer.stop_reason,
        "seconds": round(seconds, 2),
        "citations": [
            {k: getattr(c, k) for k in ("number", "chunk_id", "source", "page_start", "page_end",
                                        "cited_text", "doc_start_char", "doc_end_char", "support")}
            for c in answer.citations
        ],
        "hits": [
            {"chunk_id": h.chunk_id, "source": h.source, "start_char": h.start_char,
             "end_char": h.end_char, "scores": h.scores}
            for h in answer.hits
        ],
        "segments": [{"text": s.text, "citations": s.citations} for s in answer.segments],
        "invalid_markers": answer.invalid_markers,
        "checks": [
            {"text": c.text, "status": c.status, "issues": c.issues, "entailment": c.entailment}
            for c in checks
        ],
    }


def load_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


# -- re-checking saved answers (no LLM calls) ----------------------------------------


def segments_from_text(text: str) -> list[Segment]:
    """Rebuild an answer's segments from its text with [n] markers.

    Mirrors `InlineCitations`: each marker group cites the last sentence (or
    list item) before it, and a claim stays open across whitespace-only gaps
    between marker groups. For records saved before segments were stored.
    """
    segments: list[Segment] = []
    pending, claim, pos = "", None, 0
    for m in _MARKER_RE.finditer(text):
        between = text[pos:m.start()]
        if claim is not None and between.strip():
            segments.append(claim)
            claim = None
        pending += between
        if claim is None:
            uncited, cited = _split_last_sentence(pending)
            if uncited.strip():
                segments.append(Segment(uncited))
            else:
                cited = uncited + cited
            claim, pending = Segment(cited), ""
        claim.citations += [n for n in _numbers(m.group(1)) if n not in claim.citations]
        pos = m.end()
    rest = text[pos:]
    if claim is not None and rest.strip():
        segments.append(claim)
        claim = None
    if claim is not None:
        segments.append(claim)
    if pending + rest:
        segments.append(Segment(pending + rest))
    return segments


def rebuild(rec: dict, store) -> Answer:
    """An Answer from a saved record, with the chunk texts read back from the index."""
    hits = store.get([h["chunk_id"] for h in rec["hits"]])
    by_id = {h.chunk_id: h for h in hits}
    citations = []
    for c in rec["citations"]:
        h = by_id[c["chunk_id"]]
        citations.append(Citation(
            number=c["number"], chunk_id=c["chunk_id"], source=c["source"],
            page_start=c["page_start"], page_end=c["page_end"], section=h.section,
            cited_text=c["cited_text"], start_char=c["doc_start_char"] - h.start_char,
            end_char=c["doc_end_char"] - h.start_char, doc_start_char=c["doc_start_char"],
            doc_end_char=c["doc_end_char"], support=c["support"],
        ))
    segments = ([Segment(s["text"], s["citations"]) for s in rec["segments"]]
                if "segments" in rec else segments_from_text(rec["text"]))
    return Answer(rec["question"], rec["text"], segments, citations, hits, rec["found"],
                  rec["stop_reason"], rec["model"], invalid_markers=rec["invalid_markers"])


def recheck(records: dict[str, dict], store, validator: Validator) -> dict[str, dict]:
    """Records with their Phase 6 checks re-run by the current validator."""
    out = {}
    for qid, rec in records.items():
        if not rec["hits"]:  # stopped by the gate: nothing was checked
            out[qid] = rec
            continue
        checks = validator.validate(rebuild(rec, store)).checks
        out[qid] = {**rec, "checks": [
            {"text": c.text, "status": c.status, "issues": c.issues, "entailment": c.entailment}
            for c in checks
        ]}
    return out


def run(questions: list[Question], generator: Generator, k: int, path: Path, delay: float) -> None:
    done = load_records(path)
    todo = [q for q in questions if q.id not in done]
    print(f"{len(done)} answers saved in {path.name}, {len(todo)} to go")
    failures = 0
    with path.open("a") as out:
        for i, q in enumerate(todo, 1):
            started = time.perf_counter()
            try:
                answer = generator.answer(q.question, k=k)
            except GenerationError as e:
                failures += 1
                print(f"[{i}/{len(todo)}] {q.id} failed: {e}")
                if failures >= MAX_FAILURES:
                    print("Stopping; run the same command again later to resume.")
                    return
                time.sleep(RETRY_WAIT)
                continue
            failures = 0
            rec = record(q, answer, time.perf_counter() - started)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"[{i}/{len(todo)}] {q.id} {rec['model'] or 'no LLM call'} {rec['seconds']:.1f}s")
            if answer.hits:  # only LLM calls count against the rate limit
                time.sleep(delay)


def report(questions: list[Question], records: dict[str, dict], tag: str) -> dict:
    rows = []
    for q in questions:
        if q.id in records:
            rec = records[q.id]
            warned = sum(c["status"] == "warn" for c in rec["checks"])
            rows.append({"q": q, "rec": rec, "warned": warned, **answer_scores(q, rec)})
    models = Counter(r["rec"]["model"] or "(gate: no LLM call)" for r in rows)
    print(f"\nAnswers '{tag}': {len(rows)}/{len(questions)} questions; "
          + ", ".join(f"{m} {n}" for m, n in models.most_common()))

    def line(label: str, group: list[dict]) -> None:
        if not group:
            return
        n = len(group)
        c = Counter(r["outcome"] for r in group)
        text = f"  {label:<26} {n:>3}  correct {c['correct']:>2}  partial {c['partial']:>2}  " \
               f"wrong {c['wrong']:>2}  refused {c['refused']:>2}"
        if group[0]["q"].answerable:
            text += (f"  | cites evidence {sum(r['cites_evidence'] for r in group):>2}"
                     f"  quote on evidence {sum(r['quote_on_evidence'] for r in group):>2}"
                     f"  wrong-doc citations {sum(r['wrong_doc_citations'] for r in group):>2}")
        print(text)

    answerable = [r for r in rows if r["q"].answerable]
    for label, docs in (("syllabus", "bsccsit"), ("Attention paper", "NIPS"), ("ViT paper", "Zhai")):
        line(label, [r for r in answerable if r["q"].kind == "single"
                     and next(iter(r["q"].sources)).startswith(docs)])
    line("multi-document", [r for r in answerable if r["q"].kind == "multi"])
    line("all answerable", answerable)
    for kind in ("unanswerable", "off_topic"):
        group = [r for r in rows if r["q"].kind == kind]
        if group:
            ok = sum(r["outcome"] == "correct" for r in group)
            gated = sum(not r["rec"]["found"] and not r["rec"]["hits"] for r in group)
            print(f"  {kind:<26} {len(group):>3}  said 'not in the documents' {ok:>2}"
                  f"  (stopped by the gate: {gated})")

    good = [r for r in answerable if r["outcome"] == "correct"]
    bad = [r for r in answerable if r["outcome"] != "correct"]
    print("\nPhase 6 checks: answers with at least one warning")
    print(f"  correct answers: {sum(r['warned'] > 0 for r in good)}/{len(good)}"
          f"   partial/wrong/refused answers: {sum(r['warned'] > 0 for r in bad)}/{len(bad)}")
    llm = [r["rec"]["seconds"] for r in rows if r["rec"]["hits"]]
    if llm:
        print(f"  time per answered question: median {statistics.median(llm):.1f}s")

    print("\nNot fully correct:")
    for r in rows:
        if r["outcome"] != "correct":
            text = " ".join(r["rec"]["text"].split())
            print(f"  {r['q'].id:<9} {r['outcome']:<8} facts {r.get('facts', '-'):<4} "
                  f"warnings {r['warned']}  {text[:150]}")
    return {r["q"].id: {k: v for k, v in r.items() if k not in ("q", "rec")} for r in rows}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Answer-level evaluation on the question set.")
    parser.add_argument("--tag", required=True, help="name of this run (answers-<tag>.jsonl)")
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=config.USE_RERANKER)
    parser.add_argument("--candidates", type=int, default=config.RETRIEVE_CANDIDATES)
    parser.add_argument("--size", type=int, default=config.CHUNK_SIZE, help="chunk size in tokens")
    parser.add_argument("-k", type=int, default=config.TOP_K, help="chunks given to the LLM")
    parser.add_argument("--provider", default=config.LLM_PROVIDER)
    parser.add_argument("--no-fallback", action="store_true",
                        help="Gemini: retry later instead of switching models, so one model answers all")
    parser.add_argument("--delay", type=float, default=5.0, help="seconds between LLM calls")
    parser.add_argument("--only", nargs="+", help="question ids to run")
    parser.add_argument("--score-only", action="store_true", help="re-score saved answers")
    parser.add_argument("--recheck", action="store_true",
                        help="re-run the Phase 6 checks on saved answers, then score (no LLM calls)")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    documents, questions = load_questions(args.questions)
    pages, texts = load_documents(documents)
    issues = problems(documents, questions) + locate(questions, texts)
    if issues:
        sys.exit("Question set problems:\n" + "\n".join(issues))
    if args.only:
        questions = [q for q in questions if q.id in args.only]

    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"answers-{args.tag}.jsonl"
    if args.recheck:
        records = recheck(load_records(path), open_index(args.size, pages, Embedder()), Validator())
        scores = report(questions, records, f"{args.tag}, re-checked")
        (RUNS_DIR / f"scores-{args.tag}-recheck.json").write_text(json.dumps(scores, indent=1))
        return
    if not args.score_only:
        store = open_index(args.size, pages, Embedder())
        retriever = Retriever(store, reranker=Reranker() if args.rerank else None,
                              use_reranker=args.rerank, candidates=args.candidates)
        backend = make_backend(args.provider)
        if args.no_fallback and hasattr(backend, "fallback_models"):
            backend.fallback_models = []
        generator = Generator(retriever, backend=backend, validate=True)
        run(questions, generator, args.k, path, args.delay)
    scores = report(questions, load_records(path), args.tag)
    (RUNS_DIR / f"scores-{args.tag}.json").write_text(json.dumps(scores, indent=1))


if __name__ == "__main__":
    _main()
