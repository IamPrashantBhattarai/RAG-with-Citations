"""Query -> top-k chunks with similarity scores. (Phase 4)

Pipeline (each stage fixes a weakness of the one before):

1. Dense search (ChromaDB, bge embeddings): finds passages with the same
   meaning even when the wording differs, but can miss exact terms.
2. Keyword search (BM25, built in memory from the stored chunks): finds exact
   words, names, codes and rare terms that embeddings blur.
3. Relevance gate: if even the best dense match is below MIN_SCORE (cosine),
   the documents almost certainly don't cover the question, so nothing is
   returned instead of noise.
4. Reciprocal Rank Fusion merges both rankings into one list.
5. Optional (USE_RERANKER, off by default): a cross-encoder reranker reads the
   question and each candidate together and reorders them. It was slower and
   not more accurate in our benchmark (see ai_agents/04-retrieval.md), so it is
   kept as a switch to re-test on real documents, not as the default.

Try it:  python -m citerag.retriever "What does unit 7 cover?"
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from collections import Counter

import numpy as np

from citerag import config
from citerag.chunker import _title
from citerag.store import Hit, Store, _best_device

log = logging.getLogger(__name__)

# -- keyword search (BM25) -------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers herself him himself his how
    i if in into is it its itself just me more most my myself no nor not now of off on
    once only or other our ours ourselves out over own same she should so some such than
    that the their theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase words minus stopwords, with plurals folded ("scripts" -> "script")."""
    out = []
    for w in _TOKEN_RE.findall(text.lower()):
        if w in _STOPWORDS:
            continue
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 4 and w.endswith(("sses", "shes", "ches", "xes", "zes")):
            w = w[:-2]  # classes, approaches, boxes
        elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        out.append(w)
    return out


class BM25:
    """Okapi BM25 over a fixed list of documents.

    A document scores high when it contains the query's words, especially rare
    ones (idf), more than once (tf, saturating via k1), and without being padded
    out by lots of other text (length normalisation via b).
    """

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self._postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(doc, tf)]
        self._lengths: list[int] = []
        for i, doc in enumerate(docs):
            tokens = tokenize(doc)
            self._lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self._postings.setdefault(term, []).append((i, tf))
        n = len(docs)
        self._avg_len = (sum(self._lengths) / n) if n else 0.0
        self._idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self._postings.items()
        }

    def __len__(self) -> int:
        return len(self._lengths)

    def scores(self, query: str) -> dict[int, float]:
        """Doc index -> score, only for docs sharing at least one query term."""
        out: dict[int, float] = {}
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in self._postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self._lengths[i] / self._avg_len)
                out[i] = out.get(i, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        return sorted(self.scores(query).items(), key=lambda x: -x[1])[:k]


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: each list adds 1 / (k + rank) for every id it ranks.

    Uses ranks only, so a cosine similarity and a BM25 score (different scales)
    can be combined without calibration. k=60 is the value from the original paper.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


# -- reranking -------------------------------------------------------------------


class Reranker:
    """Cross-encoder: (question, passage) -> relevance in 0..1 (loaded lazily)."""

    def __init__(
        self,
        model_name: str = config.RERANK_MODEL,
        device: str | None = config.EMBED_DEVICE,
        batch_size: int = config.EMBED_BATCH_SIZE,
        max_tokens: int = config.EMBED_MAX_TOKENS,
    ):
        self.model_name = model_name
        self.device = device or _best_device()
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading reranker %s on %s", self.model_name, self.device)
            self._model = CrossEncoder(
                self.model_name, device=self.device, max_length=self.max_tokens
            )
        return self._model

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        import torch

        scores = self.model.predict(
            [(query, p) for p in passages],
            batch_size=self.batch_size,
            activation_fn=torch.nn.Sigmoid(),  # logits -> 0..1, comparable across queries
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype=np.float32)


# -- retriever -------------------------------------------------------------------


class Retriever:
    def __init__(
        self,
        store: Store | None = None,
        reranker: Reranker | None = None,
        use_reranker: bool = config.USE_RERANKER,
        candidates: int = config.RETRIEVE_CANDIDATES,
        min_score: float = config.MIN_SCORE,
    ):
        self.store = store or Store()
        self.reranker = (reranker or Reranker()) if use_reranker else None
        self.candidates = candidates
        self.min_score = min_score
        self._bm25: BM25 | None = None
        self._bm25_chunks: list[Hit] = []
        self._bm25_key: tuple[int, int] | None = None

    def retrieve(
        self, query: str, k: int = config.TOP_K, sources: list[str] | None = None
    ) -> list[Hit]:
        """Best chunks for a question, best first; [] if nothing is relevant enough.

        `hit.score` is the fused RRF score, or the reranker's relevance (0..1)
        when reranking is on; both only order results. `hit.scores` keeps every
        stage's score ("dense" cosine, "keyword" BM25, "rrf", "rerank").
        """
        if not query.strip() or not self.store.count():
            return []
        pool = max(self.candidates, k)
        where = {"source": {"$in": sources}} if sources else None

        dense = self.store.query(query, k=pool, where=where)
        if not dense or dense[0].score < self.min_score:
            best = dense[0].score if dense else 0.0
            log.info("best match %.3f < MIN_SCORE %.2f: no relevant chunks", best, self.min_score)
            return []
        keyword = self._keyword_search(query, pool, sources)
        fused = rrf([[h.chunk_id for h in dense], [h.chunk_id for h, _ in keyword]])
        order = sorted(fused, key=lambda cid: -fused[cid])[:pool]

        hits = {h.chunk_id: h for h in dense}
        hits.update({h.chunk_id: h for h, _ in keyword if h.chunk_id not in hits})
        dense_scores = {h.chunk_id: h.score for h in dense}
        keyword_scores = {h.chunk_id: s for h, s in keyword}
        results = []
        for cid in order:
            h = hits[cid]
            h.scores = {"rrf": fused[cid]}
            if cid in dense_scores:
                h.scores["dense"] = dense_scores[cid]
            if cid in keyword_scores:
                h.scores["keyword"] = keyword_scores[cid]
            h.score = fused[cid]
            results.append(h)

        if self.reranker is None:
            return results[:k]
        rel = self.reranker.score(query, [passage(h) for h in results])
        for h, r in zip(results, rel):
            h.scores["rerank"] = h.score = float(r)
        results.sort(key=lambda h: -h.score)
        return results[:k]

    def _keyword_search(
        self, query: str, k: int, sources: list[str] | None
    ) -> list[tuple[Hit, float]]:
        key = (self.store.version, self.store.count())
        if self._bm25 is None or key != self._bm25_key:
            self._bm25_chunks = self.store.all_chunks()
            self._bm25 = BM25([passage(h) for h in self._bm25_chunks])
            self._bm25_key = key
            log.info("keyword index built over %d chunks", len(self._bm25_chunks))
        allowed = set(sources) if sources else None
        ranked = sorted(self._bm25.scores(query).items(), key=lambda x: -x[1])
        out = []
        for i, score in ranked:
            hit = self._bm25_chunks[i]
            if allowed is None or hit.source in allowed:
                out.append((Hit(**{**hit.__dict__, "scores": {}}), score))
                if len(out) == k:
                    break
        return out


def passage(hit: Hit) -> str:
    """What keyword search and the reranker read: "title > section" + chunk text.

    Mirrors the header the chunker puts on the embedded text, so all three
    stages see the same context (document title and section name).
    """
    title = _title(hit.source)
    header = f"{title} > {hit.section}" if hit.section else title
    return f"{header}\n\n{hit.text}"


# -- CLI -------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Ask the index a question (no LLM).")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=config.TOP_K)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction,
                        default=config.USE_RERANKER, help="cross-encoder reranking")
    parser.add_argument("--source", action="append", help="limit to a file (repeatable)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for name in ("citerag", "__main__"):  # "__main__" when run with python -m
        logging.getLogger(name).setLevel(logging.INFO)

    retriever = Retriever(use_reranker=args.rerank)
    hits = retriever.retrieve(args.question, args.k, args.source)
    if not hits:
        print(f"No relevant chunks (best match below MIN_SCORE={retriever.min_score}): "
              "the documents probably don't answer this.")
        return
    for n, h in enumerate(hits, 1):
        pages = f"p{h.page_start}" + (f"-{h.page_end}" if h.page_end != h.page_start else "")
        stages = "  ".join(f"{k} {v:.3f}" for k, v in h.scores.items())
        print(f"\n{n}. {h.score:.3f}  {h.source} {pages}  [{h.section or '-'}]")
        print(f"   ({stages})")
        print("   " + " ".join(h.text.split())[:300])


if __name__ == "__main__":
    _main()
