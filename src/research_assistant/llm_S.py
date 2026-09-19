"""Provider-agnostic LLM access (Sude).

The model backend is deliberately not decided yet, so nothing above this module
names a provider. Everything that needs generation depends on `LLMClient`; the
concrete client is chosen once, here, from settings.

`StubLLM` is not a placeholder to be deleted later — it is the CI backend. A gate
that calls a paid API on every push is a gate people learn to skip, and a judge
whose outputs vary run-to-run cannot support a promote/reject decision. The stub
is deterministic, so the pipeline's plumbing is tested on every push, while the
quality numbers come from scheduled runs against a real model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from research_assistant.config_J import get_settings
from research_assistant.observability.tracing_S import KIND_LLM, span

logger = logging.getLogger(__name__)

# The schema a caller passes is the type it gets back. Returning a bare
# BaseModel hid every field from the type checker at every call site.
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMError(RuntimeError):
    pass


@runtime_checkable
class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str: ...

    def complete_json(
        self, system: str, user: str, schema: type[SchemaT], max_tokens: int = 1024
    ) -> SchemaT: ...


def _extract_json(text: str) -> dict[str, Any]:
    """Models wrap JSON in prose or fences more often than they should.

    Failing loudly here rather than returning a default matters: a judge that
    silently returns grade 0 on a parse error would depress every metric and look
    like a retrieval regression.
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"No JSON object in model output: {text[:200]!r}")
    return json.loads(candidate[start : end + 1])


class StubLLM:
    """Deterministic, offline, no network.

    Extractive by construction: it only ever returns sentences that appear in its
    input. That keeps it honest as a summariser (it cannot hallucinate, so a
    faithfulness failure in a stub run is a real plumbing bug) and makes its
    judge verdicts a pure function of lexical overlap.
    """

    model = "stub-deterministic-v1"

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        with span("llm.complete", KIND_LLM) as sp:
            sp.set(model=self.model)
            sentences = _sentences(user)
            if not sentences:
                return ""
            budget = max(1, min(len(sentences), max_tokens // 40))
            return " ".join(sentences[:budget])

    def complete_json(
        self, system: str, user: str, schema: type[SchemaT], max_tokens: int = 1024
    ) -> SchemaT:
        """Fill the schema deterministically from a hash of the prompt.

        Values are stable for identical inputs, which is what lets the CI gate
        compare two runs and attribute any difference to the code under test.
        """
        with span("llm.complete_json", KIND_LLM) as sp:
            sp.set(model=self.model, schema=schema.__name__)
            seed = int(hashlib.sha256((system + user).encode("utf-8")).hexdigest()[:8], 16)
            return schema.model_validate(_stub_payload(schema, seed, user))


class AnthropicLLM:
    """Real backend. Only constructed when `judge_backend='anthropic'`.

    Three constraints of the Claude 5 generation shape this class, and each was a
    latent failure in the first version, which had never been run:

    * **No sampling parameters.** `temperature`, `top_p` and `top_k` are removed
      on Claude 5 models and return a 400. The earlier `temperature=0.0` would
      have failed every call. The design's "temperature by role" therefore cannot
      be implemented on these models; run-to-run variance has to be *measured*
      (repeat the same query, compare verdicts) rather than set to zero. The CI
      gate's reproducibility still comes from the stub backend.
    * **Structured outputs, not JSON-by-prompt.** `messages.parse` constrains the
      response to the Pydantic schema, so malformed judge output — edge cases
      E16/E17 — stops being a parsing problem at all. The earlier version asked
      for JSON in the system prompt and regex-extracted it.
    * **Thinking is on by default and counts toward `max_tokens`.** A caller
      asking for a short summary with a 400-token cap could spend the entire
      budget thinking and receive no text. Output length is controlled in the
      prompt; the token cap is a safety ceiling, so it is floored.
    """

    # A ceiling, not a length control. Floored so adaptive thinking cannot exhaust
    # the budget before any answer text is produced; kept at the non-streaming
    # size that stays inside SDK request timeouts.
    MIN_MAX_TOKENS = 16000

    def __init__(self, model: str, api_key: str) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError("pip install '.[judge]' to use the anthropic backend") from exc
        self._anthropic = anthropic
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def _cap(self, max_tokens: int) -> int:
        return max(max_tokens, self.MIN_MAX_TOKENS)

    def _check_stop(self, response: Any) -> None:
        """Turn stop reasons that mean "no usable answer" into LLMError.

        Every caller already treats LLMError as a verdict it does not have — the
        editor escalates, triage degrades — so a refusal or a truncated response
        routes safely instead of being parsed as if it were complete.
        """
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            raise LLMError(f"model declined the request (category={category})")
        if response.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens; no complete answer")

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        with span("llm.complete", KIND_LLM) as sp:
            sp.set(model=self.model)
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=self._cap(max_tokens),
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
            except (self._anthropic.APIStatusError, self._anthropic.APIConnectionError) as exc:
                raise LLMError(f"API call failed: {exc}") from exc
            self._check_stop(msg)
            sp.set(input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens)
            return "".join(b.text for b in msg.content if b.type == "text")

    def complete_json(
        self, system: str, user: str, schema: type[SchemaT], max_tokens: int = 1024
    ) -> SchemaT:
        with span("llm.complete_json", KIND_LLM) as sp:
            sp.set(model=self.model, schema=schema.__name__)
            try:
                msg = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._cap(max_tokens),
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    output_format=schema,
                )
            except (self._anthropic.APIStatusError, self._anthropic.APIConnectionError) as exc:
                raise LLMError(f"API call failed: {exc}") from exc
            except ValidationError as exc:
                # Numeric bounds are not enforced by the API's schema subset - the
                # SDK moves them into descriptions and validates locally. An
                # out-of-range value surfaces here, and must take the same safe
                # path as any other unusable verdict rather than crash the run.
                raise LLMError(f"{schema.__name__} failed validation: {exc}") from exc
            self._check_stop(msg)
            sp.set(input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens)
            if msg.parsed_output is None:
                raise LLMError(f"no parsed output for {schema.__name__}")
            return msg.parsed_output


class GeminiLLM:
    """Real backend, Google's Gemini API.

    Added alongside `AnthropicLLM` rather than instead of it — the two are picked
    by `judge_backend`, and this project has no reason to force one team member's
    provider choice onto another. This is the one selected when no Anthropic key
    is available, which is what the free tier is for.

    Two things differ from the Anthropic path, for the opposite reason each time:

    * **Sampling parameters work here.** Claude 5 models removed `temperature`;
      Gemini did not. `docs/architecture_J.md`'s "temperature by role" decision
      is implementable again — but only partially, and said so rather than
      quietly picked one reading: every role currently shares one `get_llm()`
      singleton, so a single client only has one temperature to give. That is
      recorded as a gap, not fixed here, because fixing it means a client per
      role, which is a bigger change than "add a provider".
    * **Structured output is validated locally, not trusted from `response.parsed`.**
      `response_json_schema` constrains the model to emit syntactically valid
      JSON; this class then validates that JSON against the Pydantic schema
      itself with `model_validate_json`. Same reasoning as `AnthropicLLM`: relying
      on an SDK convenience field neither of us has read the source for court a
      version-specific failure. Validating explicitly means the failure mode is
      always the same `ValidationError` — mapped once, at the end, to `LLMError`.
    """

    # A floor, not a target. Our largest schema (FaithfulnessReply, with a
    # violations list) needs real room; a caller-specified 1024 would truncate it
    # before every violation is described.
    MIN_MAX_OUTPUT_TOKENS = 4096

    def __init__(self, model: str, api_key: str, *, temperature: float = 0.0) -> None:
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as exc:  # pragma: no cover
            raise LLMError("pip install '.[judge]' to use the gemini backend") from exc
        self._types = types
        self._errors = errors
        self.model = model
        self.temperature = temperature
        self._client = genai.Client(api_key=api_key)

    def _cap(self, max_tokens: int) -> int:
        return max(max_tokens, self.MIN_MAX_OUTPUT_TOKENS)

    def _extract_text(self, response: Any) -> str:
        """Read response text, or raise LLMError for every way there can be none.

        Every caller already treats LLMError as a verdict it does not have — the
        editor escalates, triage degrades — so a safety block or a truncated
        response routes safely instead of crashing on a missing `.text`.
        """
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None)
        if block_reason:
            raise LLMError(f"prompt blocked (reason={block_reason})")
        if not response.candidates:
            raise LLMError("no candidates returned (prompt likely blocked)")
        finish_reason = response.candidates[0].finish_reason
        if finish_reason in ("SAFETY", "RECITATION"):
            raise LLMError(f"response blocked (finish_reason={finish_reason})")
        if finish_reason == "MAX_TOKENS":
            raise LLMError("response truncated at max_output_tokens; no complete answer")
        text = response.text
        if not text:
            raise LLMError(f"empty response (finish_reason={finish_reason})")
        return text

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        with span("llm.complete", KIND_LLM) as sp:
            sp.set(model=self.model)
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=self._types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=self.temperature,
                        max_output_tokens=self._cap(max_tokens),
                    ),
                )
            except self._errors.APIError as exc:
                raise LLMError(f"API call failed: {exc}") from exc
            text = self._extract_text(response)
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                sp.set(
                    input_tokens=usage.prompt_token_count,
                    output_tokens=usage.candidates_token_count,
                )
            return text

    def complete_json(
        self, system: str, user: str, schema: type[SchemaT], max_tokens: int = 1024
    ) -> SchemaT:
        with span("llm.complete_json", KIND_LLM) as sp:
            sp.set(model=self.model, schema=schema.__name__)
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=self._types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=self.temperature,
                        max_output_tokens=self._cap(max_tokens),
                        response_mime_type="application/json",
                        response_json_schema=schema.model_json_schema(),
                    ),
                )
            except self._errors.APIError as exc:
                raise LLMError(f"API call failed: {exc}") from exc
            text = self._extract_text(response)
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                sp.set(
                    input_tokens=usage.prompt_token_count,
                    output_tokens=usage.candidates_token_count,
                )
            try:
                return schema.model_validate_json(text)
            except ValidationError as exc:
                raise LLMError(f"{schema.__name__} failed validation: {exc}") from exc


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is not None:
        return _client
    s = get_settings()
    if s.judge_backend == "anthropic":
        if not s.anthropic_api_key:
            raise LLMError("judge_backend='anthropic' but RA_ANTHROPIC_API_KEY is unset")
        _client = AnthropicLLM(s.judge_model, s.anthropic_api_key)
        logger.info("LLM backend: anthropic (%s)", s.judge_model)
    elif s.judge_backend == "gemini":
        if not s.gemini_api_key:
            raise LLMError("judge_backend='gemini' but RA_GEMINI_API_KEY is unset")
        _client = GeminiLLM(s.judge_model, s.gemini_api_key)
        logger.info("LLM backend: gemini (%s)", s.judge_model)
    else:
        _client = StubLLM()
        logger.info("LLM backend: stub (deterministic, offline)")
    return _client


def set_llm(client: LLMClient | None) -> None:
    """Test hook - inject a fake, or pass None to re-detect from settings."""
    global _client
    _client = client


# --- stub payload construction ----------------------------------------------

_SENT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


_CHUNK_REF = re.compile(r"^\[([0-9a-f]{6,})\]", re.MULTILINE)


def _cited_chunk_ids(user: str) -> list[str]:
    """Chunk ids from the passages block of a prompt, in order, deduplicated."""
    seen: list[str] = []
    for cid in _CHUNK_REF.findall(user):
        if cid not in seen:
            seen.append(cid)
    return seen


def _stub_draft(user: str) -> tuple[str, list[dict[str, str]]]:
    """A deterministic cited draft built from whatever passages are in the prompt.

    The stub has to produce something *coherent*, not merely type-correct: the
    editor decomposes the draft into claims and matches them by string equality,
    so a payload of empty strings would exercise none of the pipeline it exists
    to test. Every claim here cites a real id from the context and appears
    verbatim in the draft, which is exactly the invariant the real writer must
    also satisfy.
    """
    ids = _cited_chunk_ids(user)
    if not ids:
        return "The retrieved passages do not address this question.", []
    claims = [
        {"text": f"Passage {cid} is relevant to this question [{cid}].", "chunk_id": cid}
        for cid in ids[:3]
    ]
    return " ".join(c["text"] for c in claims), claims


def _stub_payload(schema: type[BaseModel], seed: int, user: str) -> dict[str, Any]:
    """Best-effort deterministic instance of an arbitrary contract model.

    The stub always produces a *passing* verdict and a well-formed draft. That is
    deliberate: its job is to prove the plumbing runs end to end on every push,
    not to simulate failures. Failure paths are tested by injecting a scripted
    client through `set_llm`, which is explicit about what it is producing rather
    than depending on a hash landing in a particular bucket.
    """
    draft_text, draft_claims = _stub_draft(user)
    payload: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        ann = str(field.annotation)
        if name == "grade":
            payload[name] = seed % 4
        elif name == "relevance":
            payload[name] = 3
        elif name == "confidence":
            payload[name] = round(0.5 + (seed % 50) / 100, 2)
        elif name in {"rationale", "explanation"}:
            payload[name] = f"stub verdict (seed={seed % 1000})"
        elif name in {"citation_precision", "coverage"}:
            payload[name] = 1.0
        elif name == "passed":
            payload[name] = True
        elif name == "evidence_sufficient":
            payload[name] = True
        elif name == "draft":
            payload[name] = draft_text
        elif name == "claims":
            payload[name] = draft_claims
        elif name in {"total_claims", "cited_claims", "supported_cited_claims"}:
            payload[name] = len(draft_claims)
        elif name == "query":
            # Must differ from previous attempts or the novelty check rejects it.
            # The seed already incorporates the prompt, which carries the
            # already-tried list, so this varies exactly when it needs to.
            payload[name] = f"stub reformulation {seed % 10000}"
        elif "list" in ann:
            payload[name] = []
        elif "float" in ann:
            payload[name] = 0.0
        elif "bool" in ann:
            payload[name] = False
        elif "int" in ann:
            payload[name] = 0
        elif "str" in ann:
            payload[name] = f"stub-{seed % 1000}"
    return payload
