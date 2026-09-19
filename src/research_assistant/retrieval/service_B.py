"""The single retrieval entry point: config in, `RetrievalResult` out.

Decides: the order of operations (embed query -> run both arms concurrently in
spirit, fuse, filter, rerank, truncate), that the ranker is resolved through
`reranker.registry_B.get_ranker` rather than called directly, and that the two
helpers Sude's `get_citation` / `summarize_section` tools need live here.

The registry indirection is the load-bearing seam. It is what lets the DPO model from
stage 08 replace the baseline without touching the MCP tools -- one function now
instead of a refactor in stage 09. Reasoning:
`notebooks/04_hybrid_retrieval_B.ipynb` ("The contract this stage owns").

Config: `configs/retrieval_B.yaml`. Contract: `contracts/retrieval_J.py`.

Deviation from the notebook prototype: the notebook's `hybrid_search` truncates to
`candidate_k` and stops. It has no rerank step, no registry lookup, no filtering and
no `RetrievalResult`, so it cannot satisfy the contract Sude's tools serialise. It
also applies `candidates_per_arm` as the fused truncation in the ablation cell, which
conflates two different knobs.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Protocol

from ..config_J import get_settings, load_config
from ..contracts.retrieval_J import Chunk, RetrievalRequest, RetrievalResponse, RetrievedChunk
from .bm25_B import BM25Index
from .hybrid_B import FusedCandidate, fuse
from .vector_store_B import VectorStore, VectorStoreLike


class QueryEmbedder(Protocol):
    """Anything that turns a query string into a vector."""

    def embed_query(self, query: str) -> Sequence[float]: ...


class Ranker(Protocol):
    """The shape `reranker.registry_B.get_ranker` returns."""

    name: str

    def rank(self, query: str, chunks: Sequence[Chunk]) -> list[tuple[Chunk, float]]: ...


class _SentenceTransformerEmbedder:
    """Default query embedder: the bge model named in `ingestion_B.yaml` -> `embed:`.

    The asymmetric `query_prefix` is applied here and nowhere else. Dropping it costs
    real accuracy and is the most common silent misuse of this model family
    (notebook 03, design-choice table: asymmetric prefixes).
    """

    def __init__(self) -> None:
        cfg = load_config("ingestion")["embed"]
        self._prefix: str = cfg.get("query_prefix", "")
        self._normalize: bool = bool(cfg.get("normalize", True))
        self._model_name: str = cfg["model"]
        self._device: str | None = None if cfg.get("device") == "auto" else cfg.get("device")
        self._model: Any | None = None

    def embed_query(self, query: str) -> Sequence[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
        vector = self._model.encode(
            [self._prefix + query], normalize_embeddings=self._normalize
        )[0]
        return [float(x) for x in vector]


def _resolve_ranker(name: str | None) -> Ranker:
    """Ask the registry for the active ranker.

    Imported inside the function on purpose: `reranker/registry_B.py` is being written
    in parallel by another thread, and a module-level import would make this whole
    module unimportable until that lands. It also keeps the dependency one-directional
    at import time.
    """
    from ..reranker.registry_B import get_ranker

    return get_ranker(name)


def _matches_filters(chunk: Chunk, filters: dict[str, Any] | None) -> bool:
    """Post-fusion metadata filter.

    Applied after both arms rather than pushed into them: Chroma has a `where` clause
    but `rank_bm25` has nothing equivalent, and filtering one arm but not the other
    would quietly bias fusion toward the unfiltered arm.

    `year_min`/`year_max` are range filters, not equality -- they are what the MCP
    `search_papers` tool actually sends (`contracts/mcp_tools_J.SearchPapersInput`),
    and a chunk with no recorded year fails both rather than passing by default,
    since "unknown year" is not evidence that a year filter is satisfied.
    """
    if not filters:
        return True
    meta = chunk.metadata
    for field, wanted in filters.items():
        if wanted is None:
            continue
        if field == "year_min":
            if meta.year is None or meta.year < int(wanted):
                return False
            continue
        if field == "year_max":
            if meta.year is None or meta.year > int(wanted):
                return False
            continue
        actual = getattr(meta, field, None)
        if isinstance(wanted, (list, tuple, set)):
            if actual not in wanted:
                return False
        elif isinstance(actual, str) and isinstance(wanted, str):
            if actual.casefold() != wanted.casefold():
                return False
        elif actual != wanted:
            return False
    return True


class RetrievalService:
    """Holds the two arms, the embedder and the ranker choice.

    Every collaborator is injectable, which is what makes the arms independently
    testable without a Chroma index on disk or a downloaded embedding model.
    """

    def __init__(
        self,
        *,
        vector_store: VectorStoreLike | None = None,
        bm25_index: BM25Index | None = None,
        embedder: QueryEmbedder | None = None,
        ranker: Ranker | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._config = config if config is not None else load_config("retrieval")
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._embedder = embedder
        self._ranker = ranker

    # -- lazy collaborators ------------------------------------------------

    @property
    def vector_store(self) -> VectorStoreLike:
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    @property
    def bm25_index(self) -> BM25Index:
        if self._bm25_index is None:
            self._bm25_index = BM25Index.load()
        return self._bm25_index

    @property
    def embedder(self) -> QueryEmbedder:
        if self._embedder is None:
            self._embedder = _SentenceTransformerEmbedder()
        return self._embedder

    @property
    def ranker(self) -> Ranker:
        if self._ranker is None:
            self._ranker = _resolve_ranker(self._config.get("rerank", {}).get("active"))
        return self._ranker

    # -- the arms, separately callable so they can be timed and ablated ----

    def bm25_arm(self, query: str, n: int) -> list[tuple[str, float]]:
        """Sparse candidates as `(chunk_id, bm25_score)`."""
        return self.bm25_index.search_scored(query, n)

    def vector_arm(self, query: str, n: int) -> list[tuple[str, float]]:
        """Dense candidates as `(chunk_id, distance)`, smaller distance being better."""
        return self.vector_store.search(self.embedder.embed_query(query), n)

    # -- the entry point ---------------------------------------------------

    @property
    def corpus_version(self) -> str:
        """`config_J.Settings` is the one source of truth for both stamps -- the
        gate and the MCP tools read the same `get_settings()` this delegates to,
        so a corpus bump or an embedding-model change cannot update on one side
        without the other noticing."""
        return get_settings().corpus_version

    @property
    def embedding_model(self) -> str:
        return get_settings().embedding_model

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Run both arms, fuse, filter, rerank via the registry, return the contract type.

        This is the method `mcp_server.backend_S.TrackARetrieval` binds to by name --
        renaming it breaks the seam silently (a lazy import that resolves to the
        wrong attribute raises only when a query actually runs, not at startup).
        """
        started = time.perf_counter()

        hybrid_cfg = self._config.get("hybrid", {})
        per_arm = int(hybrid_cfg.get("candidates_per_arm", 30))
        candidate_k = int(request.candidate_k)
        top_k = int(request.top_k)
        fusion = hybrid_cfg.get("fusion", "rrf")

        bm25_scored = self.bm25_arm(request.query, per_arm)
        vector_scored = self.vector_arm(request.query, per_arm)
        bm25_raw = dict(bm25_scored)
        # Dense arm returns a distance (smaller is better); stamped as a score for
        # provenance, so it is negated to the same "bigger is better" convention as
        # every other score field on the contract.
        vector_raw = {cid: -float(dist) for cid, dist in vector_scored}

        # Fuse over the full per-arm lists, then truncate: filtering before truncation
        # would otherwise let a filter shrink the candidate set below candidate_k for
        # no reason. `fuse` handles dedupe_by: chunk_id.
        fused = fuse(
            bm25_scored,
            vector_scored,
            fusion=fusion,
            rrf_k=int(hybrid_cfg.get("rrf_k", 60)),
            weights=dict(hybrid_cfg.get("weights", {}) or {}),
            candidate_k=0,  # no truncation yet
        )

        candidates: list[tuple[Chunk, FusedCandidate]] = []
        for candidate in fused:
            chunk = self._load_chunk(candidate.chunk_id)
            if chunk is None or not _matches_filters(chunk, request.filters):
                continue
            candidates.append((chunk, candidate))
            if len(candidates) >= candidate_k:
                break

        if request.use_reranker and candidates:
            ranker = self.ranker
            reranker_version = str(getattr(ranker, "name", getattr(ranker, "version", "unknown")))
            ranked = ranker.rank(request.query, [chunk for chunk, _ in candidates])
        else:
            # `use_reranker=False` (or an empty candidate set) means fusion order
            # stands; MCP callers use this to isolate a retrieval-only measurement
            # from the reranker's contribution.
            reranker_version = "none"
            ranked = [(chunk, fc.score) for chunk, fc in candidates]

        results: list[RetrievedChunk] = []
        for position, (chunk, score) in enumerate(ranked[:top_k], start=1):
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(score),
                    bm25_score=bm25_raw.get(chunk.chunk_id),
                    vector_score=vector_raw.get(chunk.chunk_id),
                    rerank_score=float(score) if request.use_reranker else None,
                    rank=position,
                )
            )

        return RetrievalResponse(
            query=request.query,
            results=results,
            corpus_version=self.corpus_version,
            embedding_model=self.embedding_model,
            reranker_version=reranker_version,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def search(
        self,
        query: str,
        candidate_k: int | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Convenience wrapper over `retrieve` for callers that predate the request
        model (own scripts, the eval harness). Not the contract entry point -- that
        is `retrieve` above."""
        hybrid_cfg = self._config.get("hybrid", {})
        rerank_cfg = self._config.get("rerank", {})
        # `RetrievalRequest.filters` is typed `dict[str, str | int | list[str]]` with
        # no `None` arm, but a caller sending `{"paper_id": None}` means "no filter on
        # this field" -- the same thing omitting the key would mean. Dropped here
        # rather than widening the contract's type for a value it should never carry.
        clean_filters = {k: v for k, v in (filters or {}).items() if v is not None}
        request = RetrievalRequest(
            query=query,
            candidate_k=int(candidate_k if candidate_k is not None else hybrid_cfg["candidate_k"]),
            top_k=int(top_k if top_k is not None else rerank_cfg["top_k"]),
            filters=clean_filters,
        )
        return self.retrieve(request)

    # -- helpers Sude's tools need ----------------------------------------

    def _load_chunk(self, chunk_id: str) -> Chunk | None:
        """Chunk by id, preferring the in-memory BM25 catalogue over a store round-trip."""
        index = self._bm25_index
        if index is not None:
            chunk = index.by_id.get(chunk_id)
            if chunk is not None:
                return chunk
        try:
            return self.vector_store.get(chunk_id)
        except KeyError:
            return None

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """One chunk by id, or None. Backs `get_citation`."""
        return self._load_chunk(chunk_id)

    def get_section(self, paper_id: str, section: str) -> list[Chunk]:
        """Every chunk of one section of one paper, in reading order.

        Backs `summarize_section`. Ordered by page then chunk_id so the summary sees
        the section as written rather than as ranked. Section matching is
        case-insensitive because headings come out of the PDF parser inconsistently
        cased.
        """
        wanted = section.casefold()
        matches = [
            chunk
            for chunk in self.bm25_index.chunks
            if chunk.metadata.paper_id == paper_id
            and (chunk.metadata.section or "").casefold() == wanted
        ]
        matches.sort(key=lambda c: (c.metadata.char_start or 0, c.chunk_id))
        return matches


# -- module-level convenience API --------------------------------------------
# One shared, lazily built service so callers do not each pay for loading the BM25
# pickle and the embedding model. Tests use `RetrievalService(...)` directly.

_DEFAULT: RetrievalService | None = None


def get_service() -> RetrievalService:
    """The process-wide default service."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = RetrievalService()
    return _DEFAULT


def reset_service() -> None:
    """Drop the cached default service. For tests and for config reloads."""
    global _DEFAULT
    _DEFAULT = None


def retrieve(request: RetrievalRequest) -> RetrievalResponse:
    """The contract entry point. `mcp_server.backend_S.TrackARetrieval` calls
    `RetrievalService().retrieve(...)` directly rather than this module-level
    function, but it is kept here too for anything that wants a plain import."""
    return get_service().retrieve(request)


def search(
    query: str,
    candidate_k: int | None = None,
    top_k: int | None = None,
    filters: dict[str, Any] | None = None,
) -> RetrievalResponse:
    """Convenience wrapper for own scripts and the eval harness. See
    `RetrievalService.search`."""
    return get_service().search(query, candidate_k=candidate_k, top_k=top_k, filters=filters)


def get_chunk(chunk_id: str) -> Chunk | None:
    """One chunk by id, or None. Backs `get_citation`."""
    return get_service().get_chunk(chunk_id)


def get_section(paper_id: str, section: str) -> list[Chunk]:
    """Every chunk of one section of one paper, in reading order. Backs `summarize_section`."""
    return get_service().get_section(paper_id, section)
