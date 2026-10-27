"""Retrieval benchmark on the Phase 8 question set. Local only: no LLM, no API key.

For each chunk size the evaluation documents get their own index under
eval/.cache/ (the app's index in .chroma/ is never touched). Every answerable
question is then run through each search method:

  dense    ChromaDB cosine search only
  keyword  BM25 only
  hybrid   both, fused with RRF (the app's default)
  rerank   hybrid, then the cross-encoder reranker

A chunk counts as a hit when it contains a whole evidence quote. The relevance
gate (MIN_SCORE) is analysed separately, from each question's best dense cosine.

Run:  python -m eval.retrieval                          # current settings
      python -m eval.retrieval --sizes 128 256 384 448   # chunk-size sweep
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from citerag import config
from citerag.chunker import Chunker, _join_pages
from citerag.loaders import Page, load_file
from citerag.retriever import Reranker, Retriever
from citerag.store import Embedder, Store
from eval.metrics import Question, covers, load_questions, locate, problems, retrieval_scores

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS = EVAL_DIR / "questions.json"
CACHE_DIR = EVAL_DIR / ".cache"
RUNS_DIR = EVAL_DIR / "runs"
METHODS = ("dense", "keyword", "hybrid", "rerank")
KS = (1, 3, 5, 8, 10)
THRESHOLDS = (0.35, 0.40, 0.425, 0.45, 0.475, 0.50, 0.55)

log = logging.getLogger(__name__)


def chunk_settings(size: int) -> tuple[int, int, int]:
    """(size, overlap, min) keeping the default 384/64/96 proportions."""
    return size, round(size / 6), size // 4


def load_documents(names: list[str]) -> tuple[list[Page], dict[str, str]]:
    """Pages of the evaluation documents, and each one's text as the chunker joins it."""
    missing = [n for n in names if not (config.DATA_DIR / n).is_file()]
    if missing:
        sys.exit(f"Missing from {config.DATA_DIR}: {', '.join(missing)}")
    pages, texts = [], {}
    for name in names:
        doc = sorted((p for p in load_file(config.DATA_DIR / name) if p.text.strip()),
                     key=lambda p: p.page)  # as Chunker.chunk_document does
        pages.extend(doc)
        texts[name] = _join_pages(doc)[0]
    return pages, texts


def open_index(size: int, pages: list[Page], embedder: Embedder) -> Store:
    """An index of `pages` at this chunk size (cached; rebuilt only when something changed)."""
    size, overlap, min_tokens = chunk_settings(size)
    store = Store(
        path=CACHE_DIR / f"index-{size}-{overlap}-{min_tokens}",
        collection="eval",
        embedder=embedder,
        chunker=Chunker(size, overlap, min_tokens),
    )
    store.index(pages, prune=True)
    return store


def search(method: str, retrievers: dict[str, Retriever], question: str, k: int) -> list:
    if method == "dense":
        return retrievers["hybrid"].store.query(question, k=k)
    if method == "keyword":
        return [h for h, _ in retrievers["hybrid"]._keyword_search(question, k, None)]
    return retrievers[method].retrieve(question, k=k)


def run_size(size: int, methods: list[str], questions: list[Question], pages: list[Page],
             embedder: Embedder, reranker: Reranker | None, candidates: int) -> dict:
    store = open_index(size, pages, embedder)
    # The gate is analysed on its own, so the retrievers here never drop a question.
    retrievers = {"hybrid": Retriever(store, use_reranker=False, candidates=candidates, min_score=0.0)}
    if "rerank" in methods:
        retrievers["rerank"] = Retriever(store, reranker=reranker, use_reranker=True,
                                         candidates=candidates, min_score=0.0)

    best = {q.id: store.query(q.question, k=1)[0].score for q in questions}
    answerable = [q for q in questions if q.answerable]
    # Evidence no single chunk contains whole (cut by a chunk boundary) can't be found by any method.
    chunks = store.all_chunks()
    unreachable = [q.id for q in answerable if not all(
        any(covers(c.source, c.start_char, c.end_char, e) for c in chunks) for e in q.evidence)]
    out = {"size": chunk_settings(size), "chunks": store.count(), "candidates": candidates,
           "best_dense": best, "unreachable": unreachable, "methods": {}}
    for method in methods:
        search(method, retrievers, "warm-up", max(KS))  # load models before timing
        per_q, started = {}, time.perf_counter()
        for q in answerable:
            hits = search(method, retrievers, q.question, max(KS))
            per_q[q.id] = {
                "scores": {k: retrieval_scores(q, hits, k) for k in KS},
                "words@5": sum(len(h.text.split()) for h in hits[:5]),
                "top": [h.chunk_id for h in hits[:5]],
            }
        seconds = (time.perf_counter() - started) / len(answerable)
        out["methods"][method] = {"per_question": per_q, "seconds": seconds}
    return out


def summary(per_q: dict, k: int) -> dict:
    rows = [r["scores"][k] for r in per_q.values()]
    return {
        "recall": statistics.mean(r["recall"] for r in rows),
        "all": statistics.mean(r["all"] for r in rows),
        "mrr": statistics.mean(r["rr"] for r in rows),
        "top_doc": statistics.mean(r["top_doc"] for r in rows),
    }


def gate_table(best: dict[str, float], questions: list[Question]) -> tuple[list[dict], dict]:
    """Per threshold: how many questions of each kind the gate would stop or let through."""
    by_kind: dict[str, list[float]] = {"answerable": [], "unanswerable": [], "off_topic": []}
    for q in questions:
        by_kind["answerable" if q.answerable else q.kind].append(best[q.id])
    return [
        {
            "threshold": t,
            "answerable_rejected": sum(s < t for s in by_kind["answerable"]),
            "unanswerable_rejected": sum(s < t for s in by_kind["unanswerable"]),
            "off_topic_accepted": sum(s >= t for s in by_kind["off_topic"]),
        }
        for t in THRESHOLDS
    ], by_kind


def report(results: list[dict], questions: list[Question]) -> None:
    n = sum(q.answerable for q in questions)
    print(f"\nRetrieval on {n} answerable questions (a hit = a chunk holding the whole evidence quote)")
    print(f"{'size':>11} {'chunks':>6}  {'method':<8} {'R@1':>5} {'R@3':>5} {'R@5':>5} {'R@8':>5}"
          f" {'all@5':>6} {'MRR':>5} {'top-doc':>7} {'words@5':>7} {'ms/q':>6}")
    for res in results:
        size = "/".join(map(str, res["size"]))
        if res["candidates"] != config.RETRIEVE_CANDIDATES:
            size += f" c{res['candidates']}"
        for method, m in res["methods"].items():
            s = {k: summary(m["per_question"], k) for k in KS}
            words = statistics.mean(r["words@5"] for r in m["per_question"].values())
            print(f"{size:>11} {res['chunks']:>6}  {method:<8} {s[1]['recall']:5.2f} {s[3]['recall']:5.2f}"
                  f" {s[5]['recall']:5.2f} {s[8]['recall']:5.2f} {s[5]['all']:6.2f} {s[10]['mrr']:5.2f}"
                  f" {s[1]['top_doc']:7.2f} {words:7.0f} {m['seconds'] * 1000:6.0f}")
        print(f"  {size}: evidence cut by a chunk boundary (no chunk holds it whole) for "
              f"{len(res['unreachable'])}: {' '.join(res['unreachable'])}")
        for method, m in res["methods"].items():
            missed = [qid for qid, r in m["per_question"].items() if not r["scores"][5]["all"]]
            print(f"  {size} {method}: evidence missing from top 5 for {len(missed)}: {' '.join(missed)}")

    gated = {tuple(r["size"]): r for r in results}  # the gate doesn't depend on the candidates
    for res in gated.values():
        rows, by_kind = gate_table(res["best_dense"], questions)
        size = "/".join(map(str, res["size"]))
        print(f"\nGate at size {size}: best dense cosine per question")
        for kind, scores in by_kind.items():
            print(f"  {kind:<13} n={len(scores):<3} min {min(scores):.3f}  median "
                  f"{statistics.median(scores):.3f}  max {max(scores):.3f}")
        print(f"  {'MIN_SCORE':>9}  answerable rejected  unanswerable rejected  off-topic accepted")
        for r in rows:
            mark = "  <- current" if abs(r["threshold"] - config.MIN_SCORE) < 1e-9 else ""
            print(f"  {r['threshold']:>9.3f}  {r['answerable_rejected']:>19}  "
                  f"{r['unanswerable_rejected']:>21}  {r['off_topic_accepted']:>18}{mark}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval benchmark on the Phase 8 question set.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[config.CHUNK_SIZE],
                        help="chunk sizes in tokens (overlap and minimum scale with it)")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--candidates", type=int, nargs="+", default=[config.RETRIEVE_CANDIDATES],
                        help="chunks each search returns for fusion and reranking")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for name in ("citerag", "eval", "__main__"):
        logging.getLogger(name).setLevel(logging.INFO)

    documents, questions = load_questions(args.questions)
    issues = problems(documents, questions)
    pages, texts = load_documents(documents)
    issues += locate(questions, texts)
    if issues:
        sys.exit("Question set problems:\n" + "\n".join(issues))

    embedder = Embedder()
    reranker = Reranker() if "rerank" in args.methods else None
    results = [run_size(s, args.methods, questions, pages, embedder, reranker, c)
               for s in args.sizes for c in args.candidates]
    report(results, questions)

    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"retrieval-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"documents": documents, "results": results}, indent=1))
    print(f"\nSaved {path.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    _main()
