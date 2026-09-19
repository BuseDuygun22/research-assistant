"""Dense arm storage: a thin, backend-agnostic wrapper over Chroma.

Decides: how a `Chunk` is laid out in the vector index, and what the rest of the
system is allowed to know about the store. Nothing outside this module imports
`chromadb`, and no Chroma object crosses the class boundary in either direction --
`upsert` takes contract `Chunk`s and plain float lists, `search` returns
`(chunk_id, distance)` tuples, `get` returns a contract `Chunk`. Swapping Chroma
for Qdrant is therefore a change to this file alone.

Reasoning is documented in `notebooks/03_embedding_vectorstore_B.ipynb`
(design-choice table: vector store, collection naming, normalisation).

Config: `configs/retrieval_B.yaml` -> `vector_store:`.

Deviation from the notebook prototype: the notebook writes only five metadata keys
(`paper_id, title, section, page_start, page_end`). That is not enough to rebuild a
`Chunk`, which also requires the rest of `ChunkMetadata`, so `get()` would have been
impossible. This module persists every `ChunkMetadata` field.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..config_J import load_config, resolve_path
from ..contracts.retrieval_J import Chunk, ChunkMetadata

# Scalar ChunkMetadata fields persisted as-is. `text` lives in the document slot, so
# it is absent here; `authors` is a list and needs JSON encoding, so it is handled
# separately in `_chunk_to_metadata`/`_metadata_to_chunk` rather than listed here.
_METADATA_FIELDS: tuple[str, ...] = (
    "paper_id",
    "title",
    "year",
    "venue",
    "section",
    "page",
    "char_start",
    "char_end",
    "source_uri",
)


@runtime_checkable
class VectorStoreLike(Protocol):
    """The dense arm as the rest of the system sees it.

    Anything satisfying this can be injected into `retrieval.service_B`, which is how
    the two arms stay independently testable without a Chroma index on disk.
    """

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...

    def search(self, embedding: Sequence[float], n: int) -> list[tuple[str, float]]: ...

    def count(self) -> int: ...

    def get(self, chunk_id: str) -> Chunk: ...

    def delete_collection(self) -> None: ...


def _chunk_to_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    """Flatten a Chunk's metadata to Chroma-safe scalars, dropping None (Chroma
    rejects nulls). `authors` is JSON-encoded since Chroma metadata values must be
    scalar, never a list."""
    meta: dict[str, str | int | float | bool] = {}
    for field in _METADATA_FIELDS:
        value = getattr(chunk.metadata, field)
        if value is not None:
            meta[field] = value
    if chunk.metadata.authors:
        meta["authors"] = json.dumps(chunk.metadata.authors)
    return meta


def _metadata_to_chunk(chunk_id: str, document: str, metadata: dict[str, Any]) -> Chunk:
    """Inverse of `_chunk_to_metadata`. Raises pydantic ValidationError on a bad row."""
    fields = dict(metadata)
    authors_raw = fields.pop("authors", None)
    authors = json.loads(authors_raw) if authors_raw else []
    return Chunk(
        chunk_id=chunk_id,
        text=document,
        metadata=ChunkMetadata(authors=authors, **fields),
    )


class VectorStore:
    """Chroma-backed dense index over `Chunk`s.

    The constructor reads `configs/retrieval_B.yaml` unless every value is passed
    explicitly, which is what lets a test point it at a temp directory.
    """

    def __init__(
        self,
        *,
        persist_dir: str | Path | None = None,
        collection: str | None = None,
        distance: str | None = None,
    ) -> None:
        if persist_dir is None or collection is None or distance is None:
            cfg = load_config("retrieval")["vector_store"]
            persist_dir = persist_dir if persist_dir is not None else cfg["persist_dir"]
            collection = collection if collection is not None else cfg["collection"]
            distance = distance if distance is not None else cfg["distance"]

        self._persist_dir = resolve_path(persist_dir)
        self._collection_name = str(collection)
        self._distance = str(distance)
        self._client: Any | None = None
        self._collection: Any | None = None

    # -- backend plumbing -------------------------------------------------

    def _get_collection(self) -> Any:
        """Open the collection lazily so constructing a VectorStore costs nothing."""
        if self._collection is None:
            import chromadb  # imported here: heavy, and only this module may touch it

            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._persist_dir))
            self._collection = self._client.get_or_create_collection(
                self._collection_name,
                metadata={"hnsw:space": self._distance},
            )
        return self._collection

    # -- public interface (imported by the ingestion pipeline) ------------

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Insert or replace `chunks` with their pre-computed `embeddings`.

        Embedding is the ingestion pipeline's job, not the store's: this class never
        loads a model, so it stays cheap to construct and trivial to fake.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} != {len(embeddings)}"
            )
        if not chunks:
            return
        self._get_collection().upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[list(map(float, v)) for v in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[_chunk_to_metadata(c) for c in chunks],
        )

    def search(self, embedding: Sequence[float], n: int) -> list[tuple[str, float]]:
        """Nearest `n` chunk ids with their distances, closest first.

        Distance, not similarity: smaller is better. The fusion layer converts.
        """
        if n <= 0:
            return []
        res = self._get_collection().query(
            query_embeddings=[list(map(float, embedding))],
            n_results=n,
        )
        ids: list[str] = list(res["ids"][0]) if res.get("ids") else []
        distances = res.get("distances")
        dists: list[float] = list(distances[0]) if distances else [0.0] * len(ids)
        return [(cid, float(d)) for cid, d in zip(ids, dists, strict=True)]

    def count(self) -> int:
        """Number of chunks in the collection."""
        return int(self._get_collection().count())

    def get(self, chunk_id: str) -> Chunk:
        """Rebuild one `Chunk` from the index. Raises KeyError if it is not there."""
        res = self._get_collection().get(ids=[chunk_id], include=["documents", "metadatas"])
        ids = list(res.get("ids") or [])
        if not ids:
            raise KeyError(chunk_id)
        documents = res.get("documents") or [""]
        metadatas = res.get("metadatas") or [{}]
        return _metadata_to_chunk(ids[0], documents[0] or "", dict(metadatas[0] or {}))

    def delete_collection(self) -> None:
        """Drop the whole collection. Used to rebuild an index from scratch."""
        client = self._client
        if client is None:
            self._get_collection()
            client = self._client
        assert client is not None
        client.delete_collection(self._collection_name)
        self._collection = None


class InMemoryVectorStore:
    """A `VectorStoreLike` with no Chroma and no disk, for unit tests.

    Brute-force cosine distance over whatever was upserted. It lives in `src/` rather
    than in the test file because the ingestion thread wants it too, and because it is
    the cheapest possible proof that the interface really is backend-agnostic.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} != {len(embeddings)}"
            )
        for chunk, vector in zip(chunks, embeddings, strict=True):
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = [float(x) for x in vector]

    def search(self, embedding: Sequence[float], n: int) -> list[tuple[str, float]]:
        if n <= 0:
            return []
        query = [float(x) for x in embedding]
        scored = [
            (cid, 1.0 - _cosine(query, vec)) for cid, vec in self._vectors.items()
        ]
        scored.sort(key=lambda pair: (pair[1], pair[0]))
        return scored[:n]

    def count(self) -> int:
        return len(self._chunks)

    def get(self, chunk_id: str) -> Chunk:
        try:
            return self._chunks[chunk_id]
        except KeyError:
            raise KeyError(chunk_id) from None

    def delete_collection(self) -> None:
        self._chunks.clear()
        self._vectors.clear()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
