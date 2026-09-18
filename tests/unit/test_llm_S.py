"""Real-client request-shape tests (Sude).

No network. Both real clients are exercised against a fake SDK client that
records the exact request and returns a canned response, so a broken request
shape (a parameter one API rejects, a response field that doesn't exist) is
caught here instead of on the first paid call — or, for the free Gemini
backend, the first call that burns free-tier quota.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from research_assistant.llm_S import AnthropicLLM, GeminiLLM, LLMError

pytest.importorskip("anthropic")
genai_types = pytest.importorskip("google.genai.types")
genai_errors = pytest.importorskip("google.genai.errors")


class Verdict(BaseModel):
    grade: int


class FakeMessages:
    def __init__(self, stop_reason: str = "end_turn", parsed=None, text: str = "ok"):
        self.calls: list[dict] = []
        self.stop_reason = stop_reason
        self.parsed = parsed
        self.text = text

    def _response(self):
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            stop_details=SimpleNamespace(category="cyber"),
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            content=[
                SimpleNamespace(type="thinking"),
                SimpleNamespace(type="text", text=self.text),
            ],
            parsed_output=self.parsed,
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response()

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self._response()


def client(messages: FakeMessages) -> AnthropicLLM:
    llm = AnthropicLLM("claude-sonnet-5", api_key="test-key")
    llm._client = SimpleNamespace(messages=messages)
    return llm


def test_no_sampling_parameters_are_sent():
    """Claude 5 models reject temperature/top_p/top_k with a 400. The first
    version sent temperature=0.0 on every call and would never have worked."""
    msgs = FakeMessages(parsed=Verdict(grade=2))
    llm = client(msgs)
    llm.complete("sys", "user")
    llm.complete_json("sys", "user", Verdict)
    for call in msgs.calls:
        assert not {"temperature", "top_p", "top_k"} & call.keys()


def test_json_uses_structured_outputs_not_prompt_parsing():
    msgs = FakeMessages(parsed=Verdict(grade=3))
    out = client(msgs).complete_json("sys", "user", Verdict)
    assert out == Verdict(grade=3)
    assert msgs.calls[0]["output_format"] is Verdict
    assert "schema" not in msgs.calls[0]["system"].lower()


def test_small_token_caps_are_floored_so_thinking_cannot_consume_them():
    """Thinking counts toward max_tokens. A 400-token summary cap could be spent
    entirely on reasoning, returning no text."""
    msgs = FakeMessages()
    client(msgs).complete("sys", "user", max_tokens=400)
    assert msgs.calls[0]["max_tokens"] >= AnthropicLLM.MIN_MAX_TOKENS


def test_only_text_blocks_reach_the_caller():
    assert client(FakeMessages(text="answer")).complete("sys", "user") == "answer"


def test_refusal_becomes_llm_error():
    """Callers treat LLMError as a verdict they do not have: the editor escalates,
    triage degrades. A refusal must route that way, not be read as content."""
    with pytest.raises(LLMError, match="declined"):
        client(FakeMessages(stop_reason="refusal")).complete("sys", "user")


def test_truncation_becomes_llm_error():
    with pytest.raises(LLMError, match="truncated"):
        client(FakeMessages(stop_reason="max_tokens", parsed=Verdict(grade=1))).complete_json(
            "sys", "user", Verdict
        )


def test_missing_parsed_output_becomes_llm_error():
    with pytest.raises(LLMError):
        client(FakeMessages(parsed=None)).complete_json("sys", "user", Verdict)


def test_api_errors_become_llm_error():
    import anthropic
    import httpx2

    class Failing(FakeMessages):
        def create(self, **kwargs):
            raise anthropic.APIConnectionError(
                request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            )

    with pytest.raises(LLMError, match="API call failed"):
        client(Failing()).complete("sys", "user")


def test_out_of_range_structured_output_becomes_llm_error():
    """The API does not enforce numeric bounds; the SDK validates locally. A
    grade of 7 on a 0-3 field must route like any other unusable verdict."""
    from pydantic import Field

    class Bounded(BaseModel):
        grade: int = Field(..., ge=0, le=3)

    class Invalid(FakeMessages):
        def parse(self, **kwargs):
            Bounded.model_validate({"grade": 7})

    with pytest.raises(LLMError, match="failed validation"):
        client(Invalid()).complete_json("sys", "user", Bounded)


# --- GeminiLLM ---------------------------------------------------------------


class FakeModels:
    """Fakes `client.models`. Returns a real `GenerateContentResponse`-shaped
    SimpleNamespace, but the request `config=` is the real
    `types.GenerateContentConfig` the code under test built — inspecting it
    exercises the actual dataclass, not a guess at its field names."""

    def __init__(
        self,
        finish_reason: str = "STOP",
        block_reason: str | None = None,
        text: str = "ok",
        has_candidates: bool = True,
    ):
        self.calls: list[dict] = []
        self.finish_reason = finish_reason
        self.block_reason = block_reason
        self.text = text
        self.has_candidates = has_candidates

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        candidates = (
            [SimpleNamespace(finish_reason=self.finish_reason)]
            if self.has_candidates
            else []
        )
        return SimpleNamespace(
            prompt_feedback=SimpleNamespace(block_reason=self.block_reason),
            candidates=candidates,
            text=self.text if candidates else None,
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
        )


def gemini_client(models: FakeModels) -> GeminiLLM:
    llm = GeminiLLM("gemini-flash-latest", api_key="test-key")
    llm._client = SimpleNamespace(models=models)
    return llm


def test_gemini_sends_system_instruction_and_temperature_via_config():
    """Config, not a prompt prefix - Gemini has a real system_instruction slot,
    unlike Claude 5's removed sampling parameters."""
    models = FakeModels()
    gemini_client(models).complete("sys prompt", "user question")
    config = models.calls[0]["config"]
    assert config.system_instruction == "sys prompt"
    assert config.temperature == 0.0


def test_gemini_json_uses_response_schema_not_prompt_parsing():
    models = FakeModels(text=Verdict(grade=3).model_dump_json())
    out = gemini_client(models).complete_json("sys", "user", Verdict)
    assert out == Verdict(grade=3)
    config = models.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == Verdict.model_json_schema()
    assert "schema" not in models.calls[0]["config"].system_instruction.lower()


def test_gemini_small_token_caps_are_floored():
    models = FakeModels()
    gemini_client(models).complete("sys", "user", max_tokens=400)
    assert models.calls[0]["config"].max_output_tokens >= GeminiLLM.MIN_MAX_OUTPUT_TOKENS


def test_gemini_only_returns_text_on_a_clean_finish():
    assert gemini_client(FakeModels(text="answer")).complete("sys", "user") == "answer"


def test_gemini_safety_block_becomes_llm_error():
    with pytest.raises(LLMError, match="blocked"):
        gemini_client(FakeModels(finish_reason="SAFETY")).complete("sys", "user")


def test_gemini_recitation_block_becomes_llm_error():
    with pytest.raises(LLMError, match="blocked"):
        gemini_client(FakeModels(finish_reason="RECITATION")).complete("sys", "user")


def test_gemini_truncation_becomes_llm_error():
    with pytest.raises(LLMError, match="truncated"):
        gemini_client(FakeModels(finish_reason="MAX_TOKENS")).complete("sys", "user")


def test_gemini_blocked_prompt_becomes_llm_error():
    """No candidates at all - the whole prompt was refused before generation,
    not merely a bad response to it."""
    with pytest.raises(LLMError, match="blocked"):
        gemini_client(FakeModels(block_reason="SAFETY", has_candidates=False)).complete(
            "sys", "user"
        )


def test_gemini_empty_candidates_becomes_llm_error():
    with pytest.raises(LLMError, match="no candidates"):
        gemini_client(FakeModels(has_candidates=False)).complete("sys", "user")


def test_gemini_invalid_json_becomes_llm_error():
    """response_json_schema constrains syntax, not semantics - a value outside
    a field's allowed range still has to be caught here, same as Anthropic's
    structured output."""
    with pytest.raises(LLMError, match="failed validation"):
        gemini_client(FakeModels(text='{"grade": "not a number"}')).complete_json(
            "sys", "user", Verdict
        )


def test_gemini_api_errors_become_llm_error():
    class Failing(FakeModels):
        def generate_content(self, **kwargs):
            raise genai_errors.APIError(429, {"error": {"message": "rate limited"}})

    with pytest.raises(LLMError, match="API call failed"):
        gemini_client(Failing()).complete("sys", "user")


def test_gemini_config_is_the_real_sdk_type():
    """Guards against the request silently degrading to a plain dict if the
    google-genai import path ever changes shape."""
    models = FakeModels()
    gemini_client(models).complete("sys", "user")
    assert isinstance(models.calls[0]["config"], genai_types.GenerateContentConfig)
