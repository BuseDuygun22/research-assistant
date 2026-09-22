"""Retrieval backend adapter (Sude).

The MCP tools must not import Buse's internals — they talk to whatever satisfies
`RetrievalBackend`, which is expressed purely in `contracts.retrieval_J` types.

Two implementations live here:

* `TrackARetrieval` — a thin adapter over Buse's `retrieval.service_B`, bound
  lazily so this module imports cleanly while her file is still empty.
* `StubRetrieval` — a deterministic in-memory test double. It is NOT a second
  retrieval implementation competing with Track A; it exists so the MCP server,
  the agent graph and the CI gate are runnable and testable before ingestion
  lands, and so unit tests never need a live vector DB. It is selected only when
  Track A's service is unavailable, and it says so loudly in `corpus_version`.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from research_assistant.config_J import get_settings
from research_assistant.contracts.retrieval_J import (
    Chunk,
    ChunkMetadata,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class RetrievalBackend(Protocol):
    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse: ...

    def get_chunk(self, chunk_id: str) -> Chunk | None: ...


class TrackARetrieval:
    """Adapter over Buse's `RetrievalService`.

    Bound at call time rather than import time: while `retrieval/service_B.py` is
    empty, importing this module must still succeed so Track B can be developed
    and tested independently. That is the whole point of the seam.
    """

    def __init__(self) -> None:
        self._service = None

    def _bind(self):
        if self._service is None:
            from research_assistant.retrieval.service_B import (  # type: ignore[attr-defined]
                RetrievalService,
            )

            self._service = RetrievalService()
        return self._service

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        return self._bind().retrieve(request)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._bind().get_chunk(chunk_id)


_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class StubRetrieval:
    """Deterministic BM25-only test double over an in-memory chunk list.

    Deterministic on purpose: the CI gate compares runs, so a backend that
    reordered ties randomly would produce metric drift with no code change.
    """

    CORPUS_VERSION = "stub-corpus"

    def __init__(self, chunks: Sequence[Chunk] | None = None) -> None:
        self.chunks: list[Chunk] = list(chunks) if chunks else list(_seed_corpus())
        self._by_id = {c.chunk_id: c for c in self.chunks}
        self._docs = [_tok(c.text) for c in self.chunks]
        self._df: Counter = Counter()
        for doc in self._docs:
            self._df.update(set(doc))
        self._avg_len = (sum(len(d) for d in self._docs) / len(self._docs)) if self._docs else 0.0

    def _bm25(self, q_tokens: list[str], doc: list[str], k1: float = 1.5, b: float = 0.75) -> float:
        if not doc:
            return 0.0
        n = len(self._docs)
        tf = Counter(doc)
        score = 0.0
        for term in q_tokens:
            if term not in tf:
                continue
            df = self._df.get(term, 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            denom = tf[term] + k1 * (1 - b + b * len(doc) / (self._avg_len or 1))
            score += idf * tf[term] * (k1 + 1) / denom
        return score

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        q = _tok(request.query)
        scored: list[tuple[float, Chunk]] = []
        for chunk, doc in zip(self.chunks, self._docs, strict=True):
            if not _passes_filters(chunk, request.filters):
                continue
            scored.append((self._bm25(q, doc), chunk))
        # Tie-break on chunk_id so ordering is stable across runs.
        scored.sort(key=lambda p: (-p[0], p[1].chunk_id))
        top = scored[: request.top_k]
        return RetrievalResponse(
            query=request.query,
            results=[
                RetrievedChunk(chunk=c, score=s, bm25_score=s, rank=i + 1)
                for i, (s, c) in enumerate(top)
            ],
            corpus_version=self.CORPUS_VERSION,
            embedding_model="none(stub)",
            reranker_version="baseline",
        )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)


def _passes_filters(chunk: Chunk, filters: dict) -> bool:
    year = chunk.metadata.year
    lo, hi = filters.get("year_min"), filters.get("year_max")
    if lo is not None and (year is None or year < int(lo)):
        return False
    if hi is not None and (year is None or year > int(hi)):
        return False
    return True


def _seed_corpus() -> list[Chunk]:
    """A handful of fixed chunks so the server answers something on a cold repo.

    Replaced entirely by Track A's index; never used when it is available.
    """
    raw = [
        (
            "stub-2020-dpr",
            "Dense Passage Retrieval for Open-Domain Question Answering",
            2020,
            "Abstract",
            "Dense passage retrieval learns a dual encoder over questions and passages, "
            "outperforming BM25 on open-domain question answering benchmarks.",
        ),
        (
            "stub-2009-rrf",
            "Reciprocal Rank Fusion Outperforms Condorcet",
            2009,
            "Method",
            "Reciprocal rank fusion combines ranked lists by summing one over k plus rank, "
            "requiring no score normalisation across heterogeneous retrievers.",
        ),
        (
            "stub-2023-dpo",
            "Direct Preference Optimization",
            2023,
            "Method",
            "Direct preference optimization fine-tunes a model on pairs of chosen and "
            "rejected responses without training an explicit reward model.",
        ),
        (
            "stub-2023-ragas",
            "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
            2023,
            "Metrics",
            "RAGAS scores faithfulness by decomposing an answer into claims and checking "
            "each claim against the retrieved context.",
        ),
    ]
    chunks = []
    for paper_id, title, year, section, text in raw:
        chunks.append(
            Chunk(
                chunk_id=Chunk.derive_id(paper_id, 0, len(text), text),
                text=text,
                metadata=ChunkMetadata(
                    paper_id=paper_id,
                    title=title,
                    year=year,
                    section=section,
                    page=1,
                    char_start=0,
                    char_end=len(text),
                ),
            )
        )
    return chunks


_backend: RetrievalBackend | None = None


def get_backend() -> RetrievalBackend:
    """Prefer Track A; fall back to the stub with a warning, never silently.

    Selection probes with a real query rather than only importing. Importability
    and serveability are different questions, and the gap between them is real:
    `RetrievalService` constructs fine with no index on disk, because its
    collaborators are lazy properties. Binding on the import alone selected a
    backend that raised on the first real query - `/ready` reported `error`,
    every tool call failed, and the stub that exists for exactly this situation
    went unused.

    `readiness()` already states the rule this follows: *a backend that imports
    but cannot answer is not ready, and only a live query distinguishes the
    two*. That test belongs here too, where the choice is actually made.

    `RA_RETRIEVAL_BACKEND` pins the choice. `stub` is what the test suite uses so
    results do not depend on whether the machine running it has a built index;
    `track_a` refuses to fall back, for a deployment that must not silently
    serve the stub.
    """
    global _backend
    if _backend is not None:
        return _backend
    choice = get_settings().retrieval_backend
    if choice == "stub":
        logger.info("Retrieval backend: stub (pinned by RA_RETRIEVAL_BACKEND)")
        _backend = StubRetrieval()
        return _backend
    candidate = TrackARetrieval()
    try:
        candidate._bind()
        # The probe: cheap (top_k=1) and paid once per process, since the result
        # is cached in `_backend`. It also warms the lazy collaborators.
        candidate.retrieve(RetrievalRequest(query="backend selection probe", top_k=1))
        _backend = candidate
        logger.info("Retrieval backend: Track A service (corpus=%s)", get_settings().corpus_version)
    except Exception as exc:  # noqa: BLE001
        if choice == "track_a":
            raise
        logger.warning(
            "Track A retrieval unavailable (%s) - using StubRetrieval. "
            "Eval numbers from this backend are NOT publishable.",
            exc,
        )
        _backend = StubRetrieval()
    return _backend


def set_backend(backend: RetrievalBackend | None) -> None:
    """Test hook - inject a fixture backend, or pass None to re-detect."""
    global _backend
    _backend = backend
