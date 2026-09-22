"""Baseline (naive) ranker — Track A / Buse, promoted from notebook 09.

Why a baseline ranker exists as a *class* rather than as "do nothing"
-------------------------------------------------------------------
Stage 04 already hands the reranker a fused candidate set that is in a sensible
order (reciprocal rank fusion over the sparse and dense arms). The cheapest
correct reranker is therefore the identity: keep that order. What matters is not
the algorithm but the *shape*: the baseline exposes exactly the same interface as
a tuned cross-encoder — a `.name` and `rank(query, chunks) -> [(Chunk, score)]` —
so swapping a trained model in is a registry entry, not a code change in
`retrieval/service_B.py`.

That symmetry is also what makes the stage 06 comparison honest: baseline and
challenger are scored by the same caller, over the same candidate set, through
the same code path. If the baseline had a different signature, every measured
delta would also contain a plumbing difference.

Two modes, both config-driven (`configs/reranker_B.yaml` is the source of truth):

* ``fusion_order`` (default) — preserve the fusion ranking and emit a monotonically
  decreasing score derived from the incoming position. Zero extra latency, and it
  is the honest "no reranking" arm of the experiment.
* ``embedding`` — optionally re-score by bi-encoder cosine similarity between the
  query and each chunk. This is a *stronger* naive baseline, but it still scores
  query and chunk independently, which is precisely the limitation a cross-encoder
  is supposed to beat. `sentence-transformers` is imported lazily so that the base
  install (and the registry) never pays for it unless this mode is selected.

Scores returned here are comparable only inside one result list, which matches the
note on `ScoredChunk.score` in the joint retrieval contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from research_assistant.contracts.retrieval_J import Chunk, RetrievedChunk

__all__ = ["Ranker", "BaselineRanker", "RankedChunk"]

RankedChunk = tuple[Chunk, float]


@runtime_checkable
class Ranker(Protocol):
    """The one interface every ranker in this package satisfies.

    `retrieval/service_B.py` depends on this and on nothing else about a ranker.
    A tuned model is a drop-in swap precisely because it implements this protocol.
    """

    name: str

    def rank(self, query: str, chunks: Sequence[Chunk]) -> list[RankedChunk]:
        """Order `chunks` for `query`, best first, each with a score."""
        ...


class BaselineRanker:
    """Naive ranker: keep fusion order, or optionally re-score by embedding similarity."""

    #: Mirrors the registry `kind` for a non-trained ranker.
    kind = "similarity"

    def __init__(
        self,
        name: str = "baseline",
        *,
        mode: str = "fusion_order",
        embedding_model: str | None = None,
    ) -> None:
        if mode not in {"fusion_order", "embedding"}:
            raise ValueError(
                f"unknown baseline mode {mode!r}, expected 'fusion_order' or 'embedding'"
            )
        self.name = name
        # `contracts.retrieval_J.Reranker` (Sude's side, and anything reading the
        # shared contract) wants `.version`; the baseline has no real versioning,
        # so it mirrors `.name` rather than inventing a fake one.
        self.version = name
        self.mode = mode
        self.embedding_model = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        self._encoder: object | None = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BaselineRanker(name={self.name!r}, mode={self.mode!r})"

    # -- the interface ---------------------------------------------------------
    def rank(self, query: str, chunks: Sequence[Chunk]) -> list[RankedChunk]:
        """Return `(chunk, score)` pairs, highest score first.

        `query` is unused in ``fusion_order`` mode by design: the baseline is the
        arm that knows nothing about the query beyond what fusion already knew.
        """
        if not chunks:
            return []
        if self.mode == "embedding":
            return self._rank_by_embedding(query, chunks)
        return self._rank_by_fusion_order(chunks)

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """`contracts.retrieval_J.Reranker` conformance: the shape anything reading
        the shared contract calls, as opposed to `.rank()`, which is the shape
        `retrieval.service_B.RetrievalService` calls internally. Kept as two
        methods rather than one: `.rank()` predates the contract and changing its
        signature would touch every existing caller for no behavioural gain, while
        a no-op reranker still needs to exist *as a `Reranker`* so the registry has
        a stamped baseline to compare a tuned model against (`reranker_version`
        stays "baseline" rather than a blank string the gate would treat as equal
        to anything).
        """
        chunks = [c.chunk for c in candidates]
        ranked = self.rank(query, chunks)
        by_chunk_id = {c.chunk.chunk_id: c for c in candidates}
        results: list[RetrievedChunk] = []
        for position, (chunk, score) in enumerate(ranked[:top_k], start=1):
            origin = by_chunk_id.get(chunk.chunk_id)
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(score),
                    bm25_score=origin.bm25_score if origin else None,
                    vector_score=origin.vector_score if origin else None,
                    rerank_score=float(score),
                    rank=position,
                )
            )
        return results

    # -- modes -----------------------------------------------------------------
    @staticmethod
    def _rank_by_fusion_order(chunks: Sequence[Chunk]) -> list[RankedChunk]:
        """Identity ranking. Score is 1/(1+position) so it is strictly decreasing."""
        return [(chunk, 1.0 / (1.0 + position)) for position, chunk in enumerate(chunks)]

    def _rank_by_embedding(self, query: str, chunks: Sequence[Chunk]) -> list[RankedChunk]:
        """Cosine similarity between the query and each chunk, scored independently."""
        encoder = self._load_encoder()
        texts = [chunk.text for chunk in chunks]
        embeddings = encoder.encode(  # type: ignore[attr-defined]
            [query, *texts], normalize_embeddings=True
        )
        query_vec = embeddings[0]
        scored = [
            (chunk, float(sum(q * c for q, c in zip(query_vec, vec, strict=True))))
            for chunk, vec in zip(chunks, embeddings[1:], strict=True)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _load_encoder(self) -> object:
        """Import sentence-transformers only when the embedding mode is actually used."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on the install
                raise RuntimeError(
                    "BaselineRanker(mode='embedding') needs sentence-transformers; "
                    'install the base dependencies with `pip install -e "."` '
                    "or use mode='fusion_order'."
                ) from exc
            self._encoder = SentenceTransformer(self.embedding_model)
        return self._encoder
