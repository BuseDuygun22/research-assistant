"""Retrieval contracts (JOINT — Buse produces, Sude consumes).

These models are the seam between Track A (ingestion + hybrid retrieval) and
Track B (reranker, MCP tools, agents). Nothing outside this module may invent
its own chunk shape.

DRAFT — requires Buse's sign-off before either side builds against it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    """Provenance for a single chunk. Every field here is citable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    paper_id: str = Field(..., description="Stable corpus-local id, e.g. arXiv id.")
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    section: str | None = Field(None, description="Nearest enclosing section heading.")
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    source_uri: str | None = Field(None, description="PDF/HTML the chunk was parsed from.")


class Chunk(BaseModel):
    """An indexed unit of text. `chunk_id` MUST be deterministic.

    Determinism matters: the eval qrels label chunk_ids by hand. If a re-ingest
    reshuffles ids, every label silently rots. `derive_id` is the only sanctioned
    way to mint one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    text: str
    metadata: ChunkMetadata

    @staticmethod
    def derive_id(paper_id: str, char_start: int, char_end: int, text: str) -> str:
        payload = f"{paper_id}:{char_start}:{char_end}:{text}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]


class RetrievedChunk(BaseModel):
    """A chunk plus the scores that put it where it is.

    Scores are kept separate rather than collapsed so the eval layer can attribute
    a win to lexical, dense, or the reranker.
    """

    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    score: float = Field(..., description="Score used for the current ordering.")
    bm25_score: float | None = None
    vector_score: float | None = None
    rerank_score: float | None = None
    rank: int = Field(..., ge=1, description="1-based position in the returned list.")


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50, description="Results returned after reranking.")
    candidate_k: int = Field(20, ge=1, le=200, description="Hybrid candidate pool size.")
    filters: dict[str, str | int | list[str]] = Field(
        default_factory=dict, description="Metadata pre-filters, e.g. {'year': 2024}."
    )
    use_reranker: bool = True


class RetrievalResponse(BaseModel):
    """Carries the index fingerprint so eval runs are never compared across
    incompatible corpora or embedding models."""

    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[RetrievedChunk]
    corpus_version: str
    embedding_model: str
    reranker_version: str = Field(
        "baseline", description="'baseline' = no cross-encoder, else the model tag."
    )
    latency_ms: float | None = None


@runtime_checkable
class Reranker(Protocol):
    """The swap-in seam. Buse's `retrieve` calls this; Sude's DPO cross-encoder
    implements it. A no-op implementation that returns candidates unchanged is a
    valid Reranker, which is what keeps the baseline path honest."""

    version: str

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Return `top_k` candidates re-ordered, with `rerank_score` and `rank` set."""
        ...
