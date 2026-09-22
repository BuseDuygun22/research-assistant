"""OllamaLLM against a mocked Ollama server (no model, no network needed)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from research_assistant.llm_S import LLMError, OllamaLLM


class _Out(BaseModel):
    answer: str
    score: int


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaLLM:
    return OllamaLLM("qwen-test", transport=httpx.MockTransport(handler))


def _reply(content: str, **extra: Any) -> httpx.Response:
    return httpx.Response(200, json={"message": {"content": content}, "done": True, **extra})


def test_complete_json_sends_schema_and_options() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _reply('{"answer": "ok", "score": 3}')

    out = _client(handler).complete_json("sys", "usr", _Out, max_tokens=100)
    assert out == _Out(answer="ok", score=3)
    assert seen["model"] == "qwen-test"
    assert seen["stream"] is False
    assert seen["format"]["properties"]["score"]["type"] == "integer"
    assert seen["options"]["temperature"] == 0.0
    # Floor: a tiny max_tokens must not truncate a JSON object mid-way.
    assert seen["options"]["num_predict"] >= OllamaLLM.MIN_NUM_PREDICT


def test_json_wrapped_in_prose_and_think_block_is_recovered() -> None:
    text = '<think>hmm</think>Here you go:\n```json\n{"answer": "ok", "score": 1}\n```'
    out = _client(lambda r: _reply(text)).complete_json("s", "u", _Out)
    assert out.score == 1


def test_invalid_json_is_retried_once_with_the_error_fed_back() -> None:
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(json.loads(request.content)["messages"][1]["content"])
        if len(prompts) == 1:
            return _reply('{"answer": "ok"}')  # missing score
        return _reply('{"answer": "ok", "score": 2}')

    out = _client(handler).complete_json("s", "u", _Out)
    assert out.score == 2
    assert len(prompts) == 2
    assert "rejected" in prompts[1]


def test_two_bad_replies_raise() -> None:
    with pytest.raises(LLMError, match="failed validation"):
        _client(lambda r: _reply("not json at all")).complete_json("s", "u", _Out)


def test_truncated_output_raises_instead_of_parsing_a_cut_off_answer() -> None:
    with pytest.raises(LLMError, match="truncated"):
        _client(lambda r: _reply('{"answer": "o', done_reason="length")).complete("s", "u")


def test_missing_model_gives_a_pull_hint() -> None:
    with pytest.raises(LLMError, match="ollama pull qwen-test"):
        _client(lambda r: httpx.Response(404, json={"error": "not found"})).complete("s", "u")


def test_unreachable_server_gives_a_serve_hint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMError, match="ollama serve"):
        _client(handler).complete("s", "u")


def test_get_llm_builds_ollama_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_assistant import llm_S
    from research_assistant.config_J import get_settings

    monkeypatch.setenv("RA_JUDGE_BACKEND", "ollama")
    monkeypatch.setenv("RA_JUDGE_MODEL", "qwen2.5:7b-instruct")
    get_settings.cache_clear()
    llm_S.set_llm(None)
    try:
        client = llm_S.get_llm()
        assert isinstance(client, OllamaLLM)
        assert client.model == "qwen2.5:7b-instruct"
    finally:
        llm_S.set_llm(None)
        monkeypatch.undo()
        get_settings.cache_clear()
