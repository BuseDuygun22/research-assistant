"""`UnionRetrieval` in isolation (Sude) - the merge logic, no discovery, no network.

`retrieval/live_discovery_S.discover` is the network-and-compute-heavy half
(covered, mocked, in `tests/unit/agents/test_live_discovery_S.py`); this file
is the pure-function half: given two already-bound backends, does the merge
produce a correctly-ranked, correctly-deduplicated result.
"""

from __future__ import annotations

from research_assistant.contracts.retrieval_J import (
    Chunk,
    ChunkMetadata,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)
from research_assistant.retrieval.live_discovery_S import UnionRetrieval


def _chunk(cid: str, title: str = "T") -> Chunk:
    return Chunk(
        chunk_id=cid, text=f"text {cid}", metadata=ChunkMetadata(paper_id="p", title=title)
    )


class _Fixed:
    """A `RetrievalBackend` returning a fixed, pre-built response."""

    def __init__(self, response: RetrievalResponse, chunks: dict[str, Chunk]) -> None:
        self._response = response
        self._chunks = chunks

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        return self._response

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)


def _response(*, corpus_version: str, pairs: list[tuple[str, float]]) -> RetrievalResponse:
    return RetrievalResponse(
        query="q",
        results=[
            RetrievedChunk(chunk=_chunk(cid), score=score, rank=i)
            for i, (cid, score) in enumerate(pairs, start=1)
        ],
        corpus_version=corpus_version,
        embedding_model="m",
        reranker_version="baseline",
    )


def test_merge_re_ranks_by_score_across_both_arms():
    primary = _Fixed(
        _response(corpus_version="papers_v2", pairs=[("p1", 0.9), ("p2", 0.5)]),
        {"p1": _chunk("p1"), "p2": _chunk("p2")},
    )
    secondary = _Fixed(
        _response(corpus_version="live-x", pairs=[("l1", 0.95), ("l2", 0.1)]),
        {"l1": _chunk("l1"), "l2": _chunk("l2")},
    )
    union = UnionRetrieval(primary, secondary)

    out = union.retrieve(RetrievalRequest(query="q", top_k=3))

    assert [r.chunk.chunk_id for r in out.results] == ["l1", "p1", "p2"]
    assert [r.rank for r in out.results] == [1, 2, 3]  # re-numbered, not carried over
    assert out.corpus_version == "papers_v2+live"


def test_merge_respects_top_k():
    primary = _Fixed(
        _response(corpus_version="papers_v2", pairs=[("p1", 0.9)]),
        {"p1": _chunk("p1")},
    )
    secondary = _Fixed(
        _response(corpus_version="live-x", pairs=[("l1", 0.8), ("l2", 0.7)]),
        {"l1": _chunk("l1"), "l2": _chunk("l2")},
    )
    union = UnionRetrieval(primary, secondary)

    out = union.retrieve(RetrievalRequest(query="q", top_k=2))
    assert len(out.results) == 2
    assert [r.chunk.chunk_id for r in out.results] == ["p1", "l1"]


def test_the_same_chunk_id_in_both_arms_is_not_duplicated():
    """A paper live-discovery re-finds that is already in the fixed corpus
    (should be rare after `_known_arxiv_ids` filtering, but the merge itself
    must be safe either way) must appear once, not twice."""
    shared = _response(corpus_version="papers_v2", pairs=[("shared", 0.6)])
    primary = _Fixed(shared, {"shared": _chunk("shared")})
    secondary = _Fixed(
        _response(corpus_version="live-x", pairs=[("shared", 0.9)]), {"shared": _chunk("shared")}
    )
    union = UnionRetrieval(primary, secondary)

    out = union.retrieve(RetrievalRequest(query="q", top_k=5))
    assert [r.chunk.chunk_id for r in out.results] == ["shared"]
    assert out.results[0].score == 0.9  # the higher-scoring copy wins


def test_get_chunk_checks_both_arms():
    primary = _Fixed(_response(corpus_version="a", pairs=[]), {"p1": _chunk("p1")})
    secondary = _Fixed(_response(corpus_version="b", pairs=[]), {"l1": _chunk("l1")})
    union = UnionRetrieval(primary, secondary)

    assert union.get_chunk("p1") is not None
    assert union.get_chunk("l1") is not None
    assert union.get_chunk("nope") is None
