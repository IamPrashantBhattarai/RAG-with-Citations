import numpy as np

from citerag.loaders import Page
from citerag.retriever import BM25, Retriever, passage, rrf, tokenize
from tests.test_store import make_store


class FakeReranker:
    """Relevance = share of query words found in the passage."""

    def score(self, query, passages):
        q = set(tokenize(query))
        return np.array([len(q & set(tokenize(p))) / len(q) for p in passages], dtype=np.float32)


DOCS = [
    Page("Magma rises through the crust and erupts as lava from volcanoes.", "geo.md", 1),
    Page("Photosynthesis turns sunlight and water into sugar inside leaves.", "bio.pdf", 1),
    Page("Error code XJ-4417 means the pump pressure sensor failed.", "manual.txt", 1),
]


def make_retriever(tmp_path, reranker=True, min_score=0.0, docs=DOCS):
    store = make_store(tmp_path)
    store.index(docs)
    return Retriever(
        store,
        reranker=FakeReranker() if reranker else None,
        use_reranker=reranker,
        candidates=10,
        min_score=min_score,
    )


# -- building blocks --------------------------------------------------------------


def test_tokenize_drops_stopwords_and_folds_plurals():
    assert tokenize("The Scripts of the Libraries, and its classes") == [
        "script", "library", "class",
    ]


def test_bm25_prefers_rare_terms_and_ignores_unrelated_docs():
    bm = BM25(["the pump failed", "the pump works fine", "a pump and code XJ4417"])
    ranked = bm.top("pump XJ4417", 3)
    assert ranked[0][0] == 2  # only doc with the rare term
    assert bm.scores("volcano") == {}


def test_bm25_normalizes_for_length():
    short = "pump failure"
    long = "pump failure " + " ".join(f"filler{i}" for i in range(50))
    bm = BM25([long, short])
    assert bm.top("pump failure", 1)[0][0] == 1


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([["a", "b", "c"], ["b", "d", "a"]])
    assert max(fused, key=fused.get) == "b"  # 2nd + 1st beats 1st + 3rd
    assert set(fused) == {"a", "b", "c", "d"}


# -- retriever ----------------------------------------------------------------------


def test_retrieve_finds_exact_code_with_all_stage_scores(tmp_path):
    hits = make_retriever(tmp_path).retrieve("what does XJ-4417 mean", k=1)
    assert hits[0].source == "manual.txt"
    assert {"rrf", "keyword", "rerank"} <= set(hits[0].scores)
    assert hits[0].score == hits[0].scores["rerank"]


def test_reranker_decides_final_order(tmp_path):
    hits = make_retriever(tmp_path).retrieve("sunlight water sugar leaves", k=3)
    assert hits[0].source == "bio.pdf"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_weak_best_match_means_no_answer(tmp_path):
    r = make_retriever(tmp_path, min_score=0.3)
    assert r.retrieve("stock market interest rates") == []
    assert r.retrieve("lava magma volcanoes")[0].source == "geo.md"


def test_reranker_only_reorders_never_drops(tmp_path):
    hits = make_retriever(tmp_path).retrieve("lava", k=3)
    assert len(hits) == 3  # unrelated chunks are ranked last, not removed


def test_without_reranker_returns_fused_order(tmp_path):
    hits = make_retriever(tmp_path, reranker=False).retrieve("pump sensor", k=2)
    assert hits and hits[0].source == "manual.txt"
    assert "rerank" not in hits[0].scores and hits[0].score == hits[0].scores["rrf"]


def test_sources_filter_applies_to_both_searches(tmp_path):
    hits = make_retriever(tmp_path).retrieve("lava pump sugar", k=5, sources=["bio.pdf"])
    assert hits and {h.source for h in hits} == {"bio.pdf"}


def test_keyword_index_rebuilds_after_new_documents(tmp_path):
    r = make_retriever(tmp_path)
    r.retrieve("lava")
    r.store.index([Page("Glaciers carve valleys over thousands of years.", "ice.md", 1)])
    hits = r.retrieve("glaciers valleys", k=1)
    assert hits[0].source == "ice.md" and "keyword" in hits[0].scores


def test_empty_query_or_empty_index_returns_nothing(tmp_path):
    assert make_retriever(tmp_path).retrieve("   ") == []
    assert make_retriever(tmp_path / "empty", docs=[]).retrieve("lava") == []


def test_passage_carries_title_and_section(tmp_path):
    hit = make_retriever(tmp_path).store.all_chunks()[0]
    hit.section = "Eruptions"
    assert passage(hit).startswith(f"{hit.source.split('.')[0]} > Eruptions\n\n")
