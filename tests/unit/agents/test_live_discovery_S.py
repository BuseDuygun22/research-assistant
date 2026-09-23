"""Live discovery, wired into the researcher node (Sude).

`retrieval/live_discovery_S.discover` does real network I/O and a real
parse/chunk/embed pass - not something a unit test should ever exercise. Every
test here mocks it at the source (`live_discovery_S.discover`), so what is
actually under test is the *wiring*: does the researcher call discovery only
when routed there, does a successful discovery's evidence actually reach the
draft, does a failed one degrade without crashing. One real, unmocked,
end-to-end run against the live arXiv API and the real corpus was run by hand
to prove the whole thing works outside the test suite (not reproducible in CI:
real network, ~30-90s, non-deterministic which papers arXiv returns).
"""

from __future__ import annotations

import pytest

from research_assistant.agents.nodes.researcher_S import researcher
from research_assistant.agents.routing_S import Budget, RoutingDecision
from research_assistant.agents.state_S import AgentState
from research_assistant.contracts.retrieval_J import Chunk, ChunkMetadata
from research_assistant.llm_S import set_llm
from research_assistant.mcp_server.backend_S import StubRetrieval, set_backend
from research_assistant.retrieval.arxiv_source_B import ArxivCandidate
from research_assistant.retrieval.live_discovery_S import DiscoveryOutcome


@pytest.fixture(autouse=True)
def reset():
    yield
    set_llm(None)
    set_backend(None)


def _discover_route_state(
    question: str = "a question the fixed corpus cannot answer",
) -> AgentState:
    """A state whose last decision is exactly what the router produces when
    local retrieval is exhausted and live discovery is available - the only
    condition under which `researcher()` should attempt it."""
    budget = Budget(re_retrievals_used=2, max_re_retrievals=2, max_live_discoveries=1)
    decision = RoutingDecision("discover", "live_discovery_attempt", "test", budget)
    state = AgentState(run_id="test-run", question=question, budget=budget)
    return state.apply("editor", decisions=(decision,), budget=budget)


def _candidate(arxiv_id: str = "2401.00001") -> ArxivCandidate:
    return ArxivCandidate(
        arxiv_id=arxiv_id,
        title="A Live-Discovered Paper",
        summary="...",
        year=2024,
        category="cs.LG",
        pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
    )


def _live_chunk() -> Chunk:
    return Chunk(
        chunk_id="live-chunk-1",
        text="A fact that only exists in the live-discovered paper.",
        metadata=ChunkMetadata(
            paper_id="livepaper1",
            title="A Live-Discovered Paper",
            venue="arXiv:2401.00001 (cs.LG) [live discovery]",
        ),
    )


class _FakeLiveBackend:
    """Stands in for the `RetrievalService` a real discovery would build."""

    def __init__(self, chunk: Chunk) -> None:
        self._chunk = chunk

    def retrieve(self, request):
        from research_assistant.contracts.retrieval_J import RetrievalResponse, RetrievedChunk

        return RetrievalResponse(
            query=request.query,
            results=[RetrievedChunk(chunk=self._chunk, score=0.99, rank=1)],
            corpus_version="live-test",
            embedding_model="test",
        )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunk if chunk_id == self._chunk.chunk_id else None


def test_researcher_only_attempts_discovery_when_routed_there(monkeypatch: pytest.MonkeyPatch):
    """The normal path (no decisions yet, or the last route was something
    else) must never call discovery - it is real network I/O and must not run
    on every question, only the one round the router explicitly chose it for."""
    called = False

    def fake_discover(*args, **kwargs):
        nonlocal called
        called = True
        return DiscoveryOutcome(papers=[], backend=None)

    monkeypatch.setattr("research_assistant.retrieval.live_discovery_S.discover", fake_discover)
    set_backend(StubRetrieval())

    researcher(AgentState(run_id="r", question="an ordinary question"))
    assert not called, "discovery must not run on the normal path"


def test_successful_discovery_unions_into_the_backend_and_is_retrievable(
    monkeypatch: pytest.MonkeyPatch,
):
    """The point of the whole feature: when discovery finds something, its
    evidence must actually become retrievable for the rest of the run, not
    just logged and discarded."""
    live_chunk = _live_chunk()

    def fake_discover(query, *, run_id, max_papers):
        return DiscoveryOutcome(papers=[_candidate()], backend=_FakeLiveBackend(live_chunk))

    monkeypatch.setattr("research_assistant.retrieval.live_discovery_S.discover", fake_discover)
    set_backend(StubRetrieval())

    state = researcher(_discover_route_state())

    assert live_chunk.chunk_id in {r.chunk.chunk_id for r in state.evidence}
    from research_assistant.mcp_server.backend_S import get_backend

    assert get_backend().get_chunk(live_chunk.chunk_id) == live_chunk


def test_discovery_finding_nothing_falls_through_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
):
    """A network failure or an empty result set must not take the run down -
    the researcher just proceeds with whatever backend was already bound."""

    def fake_discover(query, *, run_id, max_papers):
        return DiscoveryOutcome(papers=[], backend=None, error="no new candidates found")

    monkeypatch.setattr("research_assistant.retrieval.live_discovery_S.discover", fake_discover)
    stub = StubRetrieval()
    set_backend(stub)

    state = researcher(_discover_route_state())  # must not raise
    assert state.queries_issued  # the normal search still ran

    from research_assistant.mcp_server.backend_S import get_backend

    assert get_backend() is stub  # the global backend was not swapped for nothing


def test_discovery_indexing_failure_degrades_gracefully(monkeypatch: pytest.MonkeyPatch):
    """A candidate that fails to parse/chunk/embed must not crash the run -
    `live_discovery_S.discover` itself catches this and returns backend=None;
    this proves the researcher's caller side handles that outcome the same way
    it handles "found nothing"."""

    def fake_discover(query, *, run_id, max_papers):
        return DiscoveryOutcome(
            papers=[_candidate()], backend=None, error="no chunks produced from the downloaded PDFs"
        )

    monkeypatch.setattr("research_assistant.retrieval.live_discovery_S.discover", fake_discover)
    set_backend(StubRetrieval())

    state = researcher(_discover_route_state())  # must not raise
    assert state.is_terminal is False
