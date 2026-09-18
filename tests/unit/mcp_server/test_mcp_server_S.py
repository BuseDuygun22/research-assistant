"""Tool-layer tests (Sude).

The property under test everywhere: a tool never raises across the boundary, and
every error it returns tells the agent what to do next. An agent that receives a
stack trace cannot recover; one that receives a remediation can.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from research_assistant.contracts.retrieval_J import RetrievalRequest, RetrievalResponse
from research_assistant.mcp_server.api_S import app
from research_assistant.mcp_server.backend_S import get_backend, set_backend
from research_assistant.mcp_server.deadline_S import ToolTimeout, with_deadline
from research_assistant.mcp_server.health_S import readiness


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    set_backend(None)


def a_chunk_id(client) -> str:
    r = client.post("/tools/search_papers", json={"query": "retrieval", "top_k": 1}).json()
    return r["results"][0]["chunk"]["chunk_id"]


# --- health / readiness ------------------------------------------------------


def test_liveness_and_readiness_are_separate(client):
    """Collapsing them would let a deploy go green with no index."""
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["status"] in ("ok", "degraded")


def test_stub_backend_reports_degraded():
    """Stub numbers must never be mistakable for real ones."""
    status = readiness()
    if type(get_backend()).__name__ == "StubRetrieval":
        assert status.status == "degraded"
        assert "not publishable" in status.detail


# --- errors are recoverable --------------------------------------------------


def test_tools_return_errors_rather_than_raising(client):
    r = client.post("/tools/get_citation", json={"chunk_id": "nope"})
    assert r.status_code == 200
    assert r.json()["error"]["code"] == "chunk_not_found"


def test_every_error_carries_a_remediation(client):
    """`retryable` says a retry is permitted; `remediation` says which retry is
    worth making."""
    for payload, path in (
        ({"chunk_id": "nope"}, "/tools/get_citation"),
        ({"chunk_ids": ["nope"]}, "/tools/summarize_section"),
        ({"query": "zzzz-nonexistent-qqqq", "year_min": 2999}, "/tools/search_papers"),
    ):
        err = client.post(path, json=payload).json()["error"]
        assert err["remediation"], f"{path} returned an error with no remediation"


def test_no_results_remediation_mentions_the_filter_when_one_is_set(client):
    err = client.post(
        "/tools/search_papers", json={"query": "zzzz-nonexistent-qqqq", "year_min": 2999}
    ).json()["error"]
    assert "year filter" in err["remediation"]


def test_partial_summary_reports_what_is_missing(client):
    """Previously the missing ids vanished, letting the writer cite a section it
    never saw summarised."""
    cid = a_chunk_id(client)
    out = client.post(
        "/tools/summarize_section", json={"chunk_ids": [cid, "bogus"]}
    ).json()
    assert out["summary"]
    assert out["error"]["partial"] is True
    assert "bogus" in out["error"]["message"]
    assert out["grounded_in"] == [cid]


def test_full_success_has_no_error(client):
    cid = a_chunk_id(client)
    out = client.post("/tools/summarize_section", json={"chunk_ids": [cid]}).json()
    assert out["error"] is None


# --- citations ---------------------------------------------------------------


def test_citation_returns_a_verbatim_span(client):
    cid = a_chunk_id(client)
    out = client.post("/tools/get_citation", json={"chunk_id": cid, "style": "apa"}).json()
    assert out["quote"]
    assert out["formatted"]
    assert out["paper_id"]


def test_citation_rejects_an_unknown_style(client):
    """Schema-level rejection: a bad style is a caller bug, not a tool error."""
    assert client.post(
        "/tools/get_citation", json={"chunk_id": "x", "style": "chicago"}
    ).status_code == 422


# --- deadlines (E34) ---------------------------------------------------------


def test_deadline_lets_a_fast_call_through():
    assert with_deadline("fast", 5.0, lambda: "done") == "done"


def test_deadline_raises_on_a_hang():
    def hang():
        time.sleep(2.0)
        return "never"

    with pytest.raises(ToolTimeout) as exc:
        with_deadline("hang", 0.1, hang)
    assert exc.value.seconds == 0.1
    assert "hang" in str(exc.value)


def test_zero_deadline_disables_the_check():
    assert with_deadline("unbounded", 0.0, lambda: "done") == "done"


def test_a_hanging_backend_becomes_a_retryable_tool_error(client, monkeypatch):
    """The whole point: a hang is converted into something the router can see.
    Left alone it consumes no budget, trips no escalation, and just stops."""

    class HangingBackend:
        def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
            time.sleep(3.0)
            raise AssertionError("unreachable")

        def get_chunk(self, chunk_id: str):
            return None

    set_backend(HangingBackend())
    monkeypatch.setenv("RA_TOOL_DEADLINE_SECONDS", "0.2")
    from research_assistant.config_J import get_settings

    get_settings.cache_clear()
    try:
        err = client.post(
            "/tools/search_papers", json={"query": "anything", "top_k": 1}
        ).json()["error"]
        assert err["code"] == "timeout"
        assert err["retryable"] is True
        assert "smaller top_k" in err["remediation"]
    finally:
        get_settings.cache_clear()


# --- the MCP adapter itself ----------------------------------------------------
#
# Regression: server_S.py was written against the mcp 1.x FastMCP import and never
# built under the installed 2.x SDK. No test constructed the server, so it passed
# CI while being unable to start. These tests build it and speak to it.


def _server():
    from research_assistant.mcp_server.server_S import build_server

    return build_server()


def test_mcp_server_builds_under_the_installed_sdk():
    assert _server() is not None


def test_mcp_server_advertises_the_contract_tool_names():
    """The default tool name is the Python function name. The contracts define
    search_papers / get_citation / summarize_section, and an agent binds to those."""
    import asyncio

    names = sorted(t.name for t in asyncio.run(_server().list_tools()))
    assert names == ["get_citation", "search_papers", "summarize_section"]


def test_mcp_tool_call_round_trips_through_the_api_logic():
    import asyncio
    import json

    result = asyncio.run(
        _server().call_tool("search_papers", {"query": "reciprocal rank fusion", "top_k": 1})
    )
    body = json.loads(result.content[0].text)
    assert body["results"], "MCP call returned no results from the tool logic"


def test_mcp_server_exposes_readiness():
    import asyncio

    uris = [str(r.uri) for r in asyncio.run(_server().list_resources())]
    assert "health://readiness" in uris
