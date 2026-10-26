import hashlib
import re

import numpy as np
import pytest

from citerag.chunker import Chunker
from citerag.loaders import Page
from citerag.store import Embedder, EmbeddingMismatchError, Store


class FakeEmbedder:
    """Hashed bag of words: texts sharing words get similar vectors. No model download."""

    def __init__(self, model_name="fake-model", dim=256):
        self.model_name = model_name
        self.doc_prefix = ""
        self.dim = dim
        self.calls = 0  # documents embedded so far

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_documents(self, texts):
        self.calls += len(texts)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def words(texts):
    return [len(t.split()) for t in texts]


def make_store(tmp_path, embedder=None, **chunker_kw):
    chunker = Chunker(**{"chunk_size": 40, "overlap": 10, "min_tokens": 8, **chunker_kw},
                      count_tokens=words)
    return Store(tmp_path / "db", "test-index", embedder or FakeEmbedder(), chunker)


PAGES = [
    Page("# Volcanoes\n\nMagma rises through the crust and erupts as lava.", "geo.md", 1),
    Page("Photosynthesis turns sunlight, water and carbon dioxide into sugar.", "bio.pdf", 1),
    Page("Chlorophyll in the leaves absorbs mostly red and blue light.", "bio.pdf", 2),
]


def test_index_then_query_returns_closest_chunk_with_citation_fields(tmp_path):
    store = make_store(tmp_path)
    report = store.index(PAGES)
    assert sorted(report.added) == ["bio.pdf", "geo.md"]
    [hit] = store.query("where does lava and magma come from", k=1)
    assert hit.source == "geo.md" and hit.page_start == 1
    assert hit.section == "Volcanoes"
    assert "Magma rises" in hit.text
    assert 0 < hit.score <= 1


def test_missing_section_round_trips_as_none(tmp_path):
    store = make_store(tmp_path)
    store.index(PAGES)
    hit = store.query("photosynthesis sunlight sugar", k=1)[0]
    assert hit.source == "bio.pdf" and hit.section is None


def test_reindexing_unchanged_files_embeds_nothing(tmp_path):
    emb = FakeEmbedder()
    store = make_store(tmp_path, emb)
    store.index(PAGES)
    calls = emb.calls
    report = store.index(PAGES)
    assert sorted(report.unchanged) == ["bio.pdf", "geo.md"]
    assert emb.calls == calls and report.chunks == 0


def test_changed_file_is_replaced_without_stale_chunks(tmp_path):
    store = make_store(tmp_path)
    long_doc = [Page(" ".join(f"Sentence {i} about rivers." for i in range(40)), "r.txt", 1)]
    store.index(long_doc)
    assert store.sources()["r.txt"] > 1
    report = store.index([Page("Rivers are short now.", "r.txt", 1)])
    assert report.updated == ["r.txt"]
    assert store.sources() == {"r.txt": 1}


def test_changed_chunk_settings_trigger_reembedding(tmp_path):
    make_store(tmp_path).index(PAGES)
    report = make_store(tmp_path, chunk_size=30).index(PAGES)
    assert sorted(report.updated) == ["bio.pdf", "geo.md"]


def test_prune_removes_files_not_passed(tmp_path):
    store = make_store(tmp_path)
    store.index(PAGES)
    report = store.index([p for p in PAGES if p.source == "geo.md"], prune=True)
    assert report.removed == ["bio.pdf"]
    assert list(store.sources()) == ["geo.md"]


def test_index_persists_across_store_instances(tmp_path):
    make_store(tmp_path).index(PAGES)
    assert make_store(tmp_path).count() == 2


def test_opening_with_different_model_is_refused(tmp_path):
    make_store(tmp_path).index(PAGES)
    with pytest.raises(EmbeddingMismatchError, match="fake-model"):
        make_store(tmp_path, FakeEmbedder("other-model"))


def test_reset_on_open_allows_switching_models(tmp_path):
    make_store(tmp_path).index(PAGES)
    store = Store(tmp_path / "db", "test-index", FakeEmbedder("other-model"), reset=True)
    assert store.count() == 0
    store.index(PAGES)
    assert store.count() == 2


def test_reset_and_delete_source(tmp_path):
    store = make_store(tmp_path)
    store.index(PAGES)
    assert store.delete_source("bio.pdf") == 1
    assert list(store.sources()) == ["geo.md"]
    store.reset()
    assert store.count() == 0 and store.query("lava") == []


def test_query_can_filter_by_source(tmp_path):
    store = make_store(tmp_path)
    store.index(PAGES)
    hits = store.query("lava", k=5, where={"source": "bio.pdf"})
    assert hits and all(h.source == "bio.pdf" for h in hits)


def test_embedder_prefixes_queries_only_and_normalizes():
    class FakeModel:
        max_seq_length = 8192

        def encode(self, texts, **kw):
            self.seen = texts
            return np.ones((len(texts), 4)) * (0.5 if kw["normalize_embeddings"] else 9)

    emb = Embedder("m", query_prefix="Q: ", doc_prefix="", device="cpu")
    emb._model = FakeModel()
    emb.embed_documents(["doc"])
    assert emb._model.seen == ["doc"]
    vec = emb.embed_query("question")
    assert emb._model.seen == ["Q: question"]
    assert vec.dtype == np.float32 and np.allclose(vec, 0.5)
