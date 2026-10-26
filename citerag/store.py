"""Embeddings + persistent ChromaDB collection. (Phase 3)

- `Embedder` turns text into unit-length vectors with a sentence-transformers
  model. Documents and queries are embedded differently: retrieval models are
  trained with an instruction/prefix on the query side only.
- `Store` keeps one vector per chunk in ChromaDB (cosine space), with the
  citation metadata (source, pages, section, character offsets) and the exact
  chunk text alongside it.

Indexing is incremental per file: each chunk records a fingerprint of its
file's text plus the embedding/chunking settings, so re-running ingestion
skips unchanged files and fully replaces changed ones (no stale chunks).
The collection also records which embedding model built it; opening it with
a different model is refused, because vectors from two models are not
comparable and mixing them silently ruins retrieval.

Try it:
    python -m citerag.store data/            # index a folder (incremental)
    python -m citerag.store --stats          # what is indexed
    python -m citerag.store --query "..."    # nearest chunks, raw scores
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from citerag import config
from citerag.chunker import Chunk, Chunker
from citerag.loaders import Page, load_paths

log = logging.getLogger(__name__)

# HNSW index settings. Chroma's defaults (100 / 100 / 16) are tuned for speed;
# a bigger graph and search beam cost little at this scale and keep the
# approximate search effectively exact.
HNSW = {"space": "cosine", "ef_construction": 200, "ef_search": 128, "max_neighbors": 32}


class EmbeddingMismatchError(RuntimeError):
    """The collection was built with a different embedding model."""


class TextEmbedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


# -- embeddings ----------------------------------------------------------------


class Embedder:
    """sentence-transformers model -> normalized float32 vectors (loaded lazily)."""

    def __init__(
        self,
        model_name: str = config.EMBED_MODEL,
        query_prefix: str = config.EMBED_QUERY_PREFIX,
        doc_prefix: str = config.EMBED_DOC_PREFIX,
        device: str | None = config.EMBED_DEVICE,
        batch_size: int = config.EMBED_BATCH_SIZE,
        max_tokens: int = config.EMBED_MAX_TOKENS,
    ):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.device = device or _best_device()
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading embedding model %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Chunks are sized to fit; never let the model read past the cap.
            self._model.max_seq_length = min(self._model.max_seq_length, self.max_tokens)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # unit vectors: cosine == dot product
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 4 * self.batch_size,
        ).astype(np.float32)


def _best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -- vector store --------------------------------------------------------------


@dataclass
class Hit:
    chunk_id: str
    text: str
    source: str
    page_start: int
    page_end: int
    section: str | None
    start_char: int
    end_char: int
    score: float  # cosine similarity here; the retriever replaces it with its final score
    scores: dict[str, float] = field(default_factory=dict)  # per-stage scores (debugging)


@dataclass
class IndexReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    chunks: int = 0  # chunks embedded in this run


class Store:
    def __init__(
        self,
        path: str | Path = config.DB_DIR,
        collection: str = config.COLLECTION_NAME,
        embedder: TextEmbedder | None = None,
        chunker: Chunker | None = None,
        reset: bool = False,
    ):
        """`reset=True` drops the index before opening it (the way to switch models)."""
        import chromadb
        from chromadb.config import Settings

        self.embedder = embedder or Embedder()
        self.chunker = chunker or Chunker()
        self._client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        self._name = collection
        if reset:
            self._drop()
        self._col = self._open()
        self.version = 0  # bumped on every write, so caches (keyword index) know to rebuild

    def _open(self):
        col = self._client.get_or_create_collection(
            self._name,
            configuration={"hnsw": HNSW},
            metadata={"embed_model": self.embedder.model_name},
            embedding_function=None,  # we always pass vectors ourselves
        )
        built_with = (col.metadata or {}).get("embed_model")
        if built_with != self.embedder.model_name:
            raise EmbeddingMismatchError(
                f"index '{self._name}' was built with {built_with}, but the current "
                f"embedding model is {self.embedder.model_name}. Rebuild it with "
                "`python -m citerag.store data/ --reset`."
            )
        return col

    # -- writing --

    def index(self, pages: Iterable[Page], prune: bool = False) -> IndexReport:
        """Embed and store pages; unchanged files are skipped, changed ones replaced.

        With `prune`, files that are indexed but not among `pages` are removed,
        so the index mirrors exactly what was passed in.
        """
        report = IndexReport()
        by_source: dict[str, list[Page]] = {}
        for p in pages:
            by_source.setdefault(p.source, []).append(p)

        for source, doc_pages in by_source.items():
            fingerprint = self._fingerprint(doc_pages)
            existing = self._col.get(where={"source": source}, include=["metadatas"])
            old_ids = existing["ids"]
            if old_ids and all(m.get("fingerprint") == fingerprint for m in existing["metadatas"]):
                report.unchanged.append(source)
                continue
            chunks = self.chunker.chunk_document(doc_pages)
            vectors = self.embedder.embed_documents([c.embed_text for c in chunks])
            # Delete only after embedding succeeded, so a failure keeps the old version.
            if old_ids:
                self._col.delete(ids=old_ids)
            self._upsert(chunks, vectors, fingerprint)
            (report.updated if old_ids else report.added).append(source)
            report.chunks += len(chunks)
            self.version += 1
            log.info("%s %s (%d chunks)", "updated" if old_ids else "added", source, len(chunks))

        if prune:
            for source in set(self.sources()) - set(by_source):
                self.delete_source(source)
                report.removed.append(source)
                log.info("removed %s", source)
        return report

    def delete_source(self, source: str) -> int:
        ids = self._col.get(where={"source": source}, include=[])["ids"]
        if ids:
            self._col.delete(ids=ids)
            self.version += 1
        return len(ids)

    def reset(self) -> None:
        """Drop every vector (and the model lock) and start an empty collection."""
        self._drop()
        self._col = self._open()
        self.version += 1

    def _drop(self) -> None:
        if self._name in {c.name for c in self._client.list_collections()}:
            self._client.delete_collection(self._name)

    def _upsert(self, chunks: list[Chunk], vectors: np.ndarray, fingerprint: str) -> None:
        batch = self._client.get_max_batch_size()
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            self._col.upsert(
                ids=[c.id for c in part],
                embeddings=vectors[i : i + batch],
                documents=[c.text for c in part],
                metadatas=[_metadata(c, fingerprint) for c in part],
            )

    def _fingerprint(self, pages: list[Page]) -> str:
        """Changes when the file's text or any setting that shapes its vectors changes."""
        h = hashlib.sha256()
        settings = (
            self.embedder.model_name,
            getattr(self.embedder, "doc_prefix", ""),
            self.chunker.chunk_size,
            self.chunker.overlap,
            self.chunker.min_tokens,
        )
        h.update(repr(settings).encode())
        for p in sorted(pages, key=lambda p: p.page):
            h.update(f"\x00{p.page}\x00{p.text}".encode())
        return h.hexdigest()[:16]

    # -- reading --

    def query(self, text: str, k: int = config.TOP_K, where: dict | None = None) -> list[Hit]:
        """Nearest chunks by cosine similarity (raw; Phase 4 adds filtering/ranking)."""
        if not self.count():
            return []
        res = self._col.query(
            query_embeddings=[self.embedder.embed_query(text)],
            n_results=min(k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            _hit(cid, doc, meta, dist)
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def get(self, ids: list[str]) -> list[Hit]:
        """Chunks by id, in the order asked (unknown ids are skipped)."""
        if not ids:
            return []
        res = self._col.get(ids=ids, include=["documents", "metadatas"])
        found = {
            cid: _hit(cid, doc, meta, 1.0)
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
        }
        return [found[i] for i in ids if i in found]

    def all_chunks(self) -> list[Hit]:
        """Every stored chunk (score 0), e.g. to build the keyword index."""
        res = self._col.get(include=["documents", "metadatas"])
        return [
            _hit(cid, doc, meta, 1.0)
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
        ]

    def count(self) -> int:
        return self._col.count()

    def sources(self) -> dict[str, int]:
        """Indexed file -> number of chunks."""
        counts: dict[str, int] = {}
        for meta in self._col.get(include=["metadatas"])["metadatas"]:
            counts[meta["source"]] = counts.get(meta["source"], 0) + 1
        return dict(sorted(counts.items()))


def _metadata(c: Chunk, fingerprint: str) -> dict:
    meta = {
        "source": c.source,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "start_char": c.start_char,
        "end_char": c.end_char,
        "n_tokens": c.n_tokens,
        "fingerprint": fingerprint,
    }
    if c.section:  # Chroma metadata can't hold None
        meta["section"] = c.section
    return meta


def _hit(chunk_id: str, text: str, meta: dict, distance: float) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        text=text,
        source=meta["source"],
        page_start=meta["page_start"],
        page_end=meta["page_end"],
        section=meta.get("section"),
        start_char=meta["start_char"],
        end_char=meta["end_char"],
        score=1.0 - distance,
    )


# -- CLI -----------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build and inspect the vector index.")
    parser.add_argument("paths", nargs="*", type=Path, help="files or folders to index")
    parser.add_argument("--reset", action="store_true", help="drop the index first")
    parser.add_argument("--prune", action="store_true", help="remove files not in paths")
    parser.add_argument("--stats", action="store_true", help="list indexed files")
    parser.add_argument("--query", help="show the nearest chunks for a question")
    parser.add_argument("-k", type=int, default=config.TOP_K)
    args = parser.parse_args()
    # Our progress at INFO; third-party libraries (HTTP calls etc.) only on warnings.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for name in ("citerag", "__main__"):  # "__main__" when run with python -m
        logging.getLogger(name).setLevel(logging.INFO)

    store = Store(reset=args.reset)
    if args.reset:
        print("index reset")
    if args.paths:
        r = store.index(load_paths(args.paths), prune=args.prune)
        print(
            f"added {len(r.added)}, updated {len(r.updated)}, unchanged {len(r.unchanged)}, "
            f"removed {len(r.removed)} file(s); embedded {r.chunks} chunks; "
            f"index now holds {store.count()} chunks"
        )
    if args.stats:
        for source, n in store.sources().items():
            print(f"{n:6d}  {source}")
        print(f"{store.count():6d}  total chunks · model {store.embedder.model_name}")
    if args.query:
        for h in store.query(args.query, args.k):
            pages = f"p{h.page_start}" + (f"-{h.page_end}" if h.page_end != h.page_start else "")
            print(f"\n{h.score:.3f}  {h.source} {pages}  [{h.section or '-'}]")
            print("   " + " ".join(h.text.split())[:300])


if __name__ == "__main__":
    _main()
