"""The `ask` entry point end to end on the stub LLM + stub retrieval (hermetic)."""

from __future__ import annotations

import io
import json
import sys

import pytest

from research_assistant.ask_S import (
    EXIT_CODES,
    _ensure_utf8_stdout,
    ask,
    main,
    render,
    resolve_citations,
)
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


def test_ensure_utf8_stdout_reconfigures_a_legacy_codepage_stream(monkeypatch) -> None:
    """Reproduces a real crash: retrieved paper text can contain any Unicode
    character (math notation, Greek letters), and `print` on a Windows console
    or redirected file defaults to cp1252, which cannot encode most of it -
    `UnicodeEncodeError` deep inside `print` then crashes an otherwise-successful
    run. This is exactly what happened live: a chunk containing the mathematical
    italic capital H (U+1D43B) crashed `ask_S.py --json > out.txt` on Windows."""
    legacy = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", legacy)

    with pytest.raises(UnicodeEncodeError):
        print("\U0001d43b")  # reproduces the crash before the fix runs

    _ensure_utf8_stdout()
    assert sys.stdout.encoding.lower().replace("-", "") == "utf8"
    print("\U0001d43b")  # must not raise now


def test_ensure_utf8_stdout_is_a_harmless_noop_on_a_stream_without_reconfigure(
    monkeypatch,
) -> None:
    """A stream some odd runner substituted might not support `reconfigure` at
    all - the guard must not itself crash the run it exists to protect."""

    class NoReconfigure:
        pass

    monkeypatch.setattr(sys, "stdout", NoReconfigure())
    _ensure_utf8_stdout()  # must not raise
