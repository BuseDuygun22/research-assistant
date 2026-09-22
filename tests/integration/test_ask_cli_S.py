"""The `ask` entry point end to end on the stub LLM + stub retrieval (hermetic)."""

from __future__ import annotations

import json

import pytest

from research_assistant.ask_S import EXIT_CODES, ask, main, render, resolve_citations
from research_assistant.llm_S import set_llm
from research_assistant.mcp_server.backend_S import set_backend


@pytest.fixture(autouse=True)
def reset():
    yield
    set_llm(None)
    set_backend(None)


def test_ask_returns_a_complete_report() -> None:
    report = ask("How does reciprocal rank fusion compare to weighted score blending?")
    assert report["outcome"] in EXIT_CODES
    assert report["llm_backend"] == "stub"
    assert report["retrieval_backend"] == "StubRetrieval"
    json.dumps(report)  # must be serialisable as-is


def test_accepted_answer_has_resolvable_references() -> None:
    report = ask("How does reciprocal rank fusion compare to weighted score blending?")
    if report["outcome"] == "accepted":
        assert report["references"], "an accepted answer must cite something"
        assert all(r["found"] for r in report["references"])


def test_unknown_citation_is_reported_not_dropped() -> None:
    refs = resolve_citations("A claim [deadbeef0] with a made-up source.")
    assert refs == [{"chunk_id": "deadbeef0", "found": False}]


def test_render_mentions_outcome_and_backend() -> None:
    text = render(ask("Why use DPO instead of a reward model and PPO?"))
    assert "llm=stub" in text


def test_main_json_and_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--json", "What does Lost in the Middle say about context position?"])
    out = json.loads(capsys.readouterr().out)
    assert code == EXIT_CODES[out["outcome"]]


def test_mlflow_logging(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("mlflow")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    code = main(["--json", "--mlflow", "--mlflow-uri", uri, "Why use DPO instead of PPO?"])
    out = json.loads(capsys.readouterr().out)
    assert code == EXIT_CODES[out["outcome"]]
    assert out["mlflow_run_id"]
