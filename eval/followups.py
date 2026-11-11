"""Follow-up benchmark (Phase 12): does rewriting follow-ups into standalone questions help?

Each item in eval/followups.json is a two-turn chat. The follow-up is searched
three ways:

  raw      as typed ("And how big is the deposit?")
  gated    rewritten only when it looks like a follow-up (what the app does)
  always   rewritten whenever there is history

Retrieval is scored like Phase 8: is the evidence in the top 5 (reranker on)?
The follow-up is then answered the app's way (gated), and the answer must
contain the item's facts. Rewrites and answers are saved as they come
(eval/runs/followups-<tag>.json), so a re-run doesn't call the LLM again.

Run:  CITERAG_GEMINI_MODEL=gemini-3.5-flash-lite python -m eval.followups --tag lite --delay 7
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from citerag import config
from citerag.conversation import is_follow_up, standalone_question
from citerag.generator import Generator, make_backend
from citerag.retriever import Reranker, Retriever
from citerag.store import Embedder
from citerag.validator import Validator
from eval.metrics import Evidence, Question, has_fact, load_questions, locate, retrieval_scores
from eval.retrieval import EVAL_DIR, QUESTIONS, RUNS_DIR, load_documents, open_index

FOLLOWUPS = EVAL_DIR / "followups.json"
WAYS = ("raw", "gated", "always")


def main() -> None:
    parser = argparse.ArgumentParser(description="Follow-up rewriting benchmark.")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--provider", default=config.LLM_PROVIDER)
    parser.add_argument("--delay", type=float, default=5.0, help="seconds between LLM calls")
    parser.add_argument("--no-answers", action="store_true", help="retrieval only (rewrites still call the LLM)")
    parser.add_argument("--raw-answers", action="store_true",
                        help="also answer each follow-up as typed, to compare with the rewritten one")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    items = json.loads(FOLLOWUPS.read_text())["items"]
    questions = [Question(it["id"], it["question"], "single",
                          [Evidence(e["source"], e["quote"] if isinstance(e["quote"], list) else [e["quote"]])
                           for e in it["evidence"]], it["facts"]) for it in items]
    documents, _ = load_questions(QUESTIONS)
    pages, texts = load_documents(documents)
    missing = locate(questions, texts)
    if any(not e.spans for q in questions for e in q.evidence):
        raise SystemExit("Evidence not found:\n" + "\n".join(missing))
    store = open_index(config.CHUNK_SIZE, pages, Embedder())
    retriever = Retriever(store, reranker=Reranker(), use_reranker=True)
    backend = make_backend(args.provider)

    path = RUNS_DIR / f"followups-{args.tag}.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    generator = None
    for it, q in zip(items, questions):
        rec = saved.setdefault(q.id, {})
        history = [(h["question"], h["answer"]) for h in it["history"]]
        if "gated" not in rec:
            rec["gated"] = standalone_question(q.question, history, backend)
            if rec["gated"] != q.question:
                time.sleep(args.delay)
        if "always" not in rec:
            rec["always"] = (rec["gated"] if is_follow_up(q.question)
                             else standalone_question(q.question, history, backend, always=True))
            if not is_follow_up(q.question):
                time.sleep(args.delay)
        rec["raw"] = q.question
        rec["top5"] = {}
        for way in WAYS:
            hits = retriever.retrieve(rec[way], k=5)
            rec["top5"][way] = bool(retrieval_scores(q, hits, 5)["all"])
        if not args.no_answers and "answer" not in rec:
            generator = generator or Generator(retriever, backend, Validator())
            answer = generator.answer(rec["gated"])
            rec["answer"] = answer.text
            rec["warned"] = sum(c.status == "warn" for c in answer.validation.checks) if answer.validation else 0
            time.sleep(args.delay)
        if args.raw_answers and "answer_raw" not in rec:
            generator = generator or Generator(retriever, backend, Validator())
            rec["answer_raw"] = generator.answer(q.question).text
            time.sleep(args.delay)
        if "answer" in rec:
            rec["correct"] = all(has_fact(rec["answer"], group) for group in q.facts)
        if "answer_raw" in rec:
            rec["correct_raw"] = all(has_fact(rec["answer_raw"], group) for group in q.facts)
        RUNS_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(saved, indent=1, ensure_ascii=False))
        print(f"{q.id}: top5 " + " ".join(f"{w}={'Y' if rec['top5'][w] else '-'}" for w in WAYS)
              + (f" | answer {'correct' if rec.get('correct') else 'WRONG'}" if "answer" in rec else "")
              + (f" | raw answer {'correct' if rec.get('correct_raw') else 'WRONG'}" if "answer_raw" in rec else "")
              + f"\n   asked:     {q.question}\n   rewritten: {rec['gated']}", flush=True)

    n = len(questions)
    print(f"\nEvidence in the top 5 ({n} follow-ups): "
          + ", ".join(f"{w} {sum(saved[q.id]['top5'][w] for q in questions)}/{n}" for w in WAYS))
    answered = [saved[q.id] for q in questions if "answer" in saved[q.id]]
    if answered:
        print(f"Answers (gated): {sum(r['correct'] for r in answered)}/{len(answered)} correct, "
              f"{sum(r['warned'] > 0 for r in answered)} with a checker warning")
    raw = [saved[q.id] for q in questions if "answer_raw" in saved[q.id]]
    if raw:
        print(f"Answers (raw, as typed): {sum(r['correct_raw'] for r in raw)}/{len(raw)} correct")
    print(f"Saved {path.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
