"""One-off real-model smoke run (Sude).

Everything else in the repo is tested against the deterministic stub and scripted
judges. This script is the first time a real model sees the prompts, and it
checks the things only a real model can get wrong:

1. **Structured outputs work** for every schema — including the nested ones
   (violations, conflicts) and constrained fields.
2. **Claims are copied verbatim** into the draft. The faithfulness judge matches
   claims by string equality, so a paraphrased claim silently breaks repair.
3. **Citations point at retrieved evidence**, not invented ids.
4. **Triage labels the adversarial passages sensibly**: a superseded result, a
   genuine disagreement, and on-topic-but-useless evidence.
5. **Run-to-run variance.** Neither backend is guaranteed deterministic at
   temperature 0 (Claude 5 removed temperature entirely; Gemini keeps it but
   real APIs still drift a little), so the same triage is run twice and
   compared rather than assumed stable.

Checks 1–3 are hard failures (exit 1). Check 4 and 5 are judgement calls and are
reported as expected-vs-actual — a mismatch is a finding to read, not
necessarily a bug.

Provider-agnostic: whichever backend `.env` sets (`RA_JUDGE_BACKEND=gemini` or
`anthropic`) is the one this runs against. It does not force a choice — a script
that silently overrides your configured backend is how "it worked when I tested
it" and "it works for you" stop meaning the same thing.

Usage:
    python scripts/smoke_real_model_S.py

Reads the key and model from .env (RA_GEMINI_API_KEY / RA_ANTHROPIC_API_KEY,
RA_JUDGE_BACKEND, RA_JUDGE_MODEL) the same way the rest of the app does. Cost:
roughly 15–20 model calls on the small built-in corpus — free on Gemini's free
tier; a low-cost estimate is printed for paid backends. On Gemini this paces
itself to the observed 5-request/minute free-tier ceiling, so a run taking a
few minutes rather than seconds is deliberate, not a hang.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Per-million-token prices, for the printed cost estimate only. Gemini's listed
# free-tier models are $0 by definition, not a guess - see
# https://ai.google.dev/gemini-api/docs/pricing.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "gemini-flash-latest": (0.0, 0.0),
    "gemini-2.5-flash": (0.0, 0.0),
    "gemini-2.5-pro": (0.0, 0.0),
}

QUESTIONS = [
    "How does reciprocal rank fusion combine ranked lists?",
    "What does dense passage retrieval learn?",
    "Why would a system use DPO instead of a reward model with PPO?",
]


class Usage:
    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0


def _gemini_retry_delay_seconds(exc: Any, default: float = 20.0) -> float:
    """Pull the server's own suggested wait out of a 429, rather than guessing.

    Google's quota error embeds a `RetryInfo.retryDelay` (e.g. "41s") in the
    error payload precisely so a client does not have to guess a backoff.
    Falls back to `default` if the shape is not what was observed — this is
    reading an error body, not a typed field, so it must not raise on a shape
    it has not seen.
    """
    try:
        for detail in exc.details["error"]["details"]:
            if detail.get("@type", "").endswith("RetryInfo"):
                return float(detail["retryDelay"].rstrip("s")) + 2.0  # small buffer
    except Exception:  # noqa: BLE001 - parsing a foreign error body defensively
        pass
    return default


def instrument(llm: Any, usage: Usage) -> None:
    """Wrap whichever SDK client is live so every call's token usage is counted
    - and, for Gemini, so the free tier's request-per-minute quota is respected.

    The two backends expose usage under different names (Anthropic:
    `response.usage.input_tokens`; Gemini: `response.usage_metadata.
    prompt_token_count`), so this branches on the class rather than assuming one
    shape - the same reason `LLMClient` exists as a protocol instead of a
    concrete type everywhere else in the codebase.

    The pacing is specific to this script, not to `GeminiLLM` itself. The main
    app is right to let a 429 or 503 surface as `LLMError` and have the caller
    decide (triage degrades, the editor escalates) - that is the correct
    behaviour under load in production. A one-off diagnostic run driving a
    dozen-plus calls back to back against a 5-requests-per-minute free-tier
    quota is a different problem: pacing and retrying here is what lets the
    check actually finish, rather than reporting the same transient error on
    every single call.
    """
    from research_assistant.llm_S import AnthropicLLM, GeminiLLM

    if isinstance(llm, AnthropicLLM):
        target, methods = llm._client.messages, ("create", "parse")
        min_interval, max_retries = 0.0, 0

        def read_usage(response: Any) -> tuple[int, int]:
            return response.usage.input_tokens, response.usage.output_tokens

        def is_transient(exc: Exception) -> bool:
            return False

    elif isinstance(llm, GeminiLLM):
        import google.genai.errors as genai_errors

        target, methods = llm._client.models, ("generate_content",)
        # 60/5 plus margin: the observed free-tier ceiling is 5 requests/minute
        # on the current Flash model. Spacing calls at all is the difference
        # between "finishes in a few minutes" and "fails on every call".
        min_interval, max_retries = 13.0, 2

        def read_usage(response: Any) -> tuple[int, int]:
            u = response.usage_metadata
            return u.prompt_token_count, u.candidates_token_count

        def is_transient(exc: Exception) -> bool:
            # 429 = free-tier quota exceeded (has a server-suggested retryDelay).
            # 503 = "high demand", Google's own words for "usually temporary" -
            # observed in practice on this free tier and worth the same retry.
            return isinstance(exc, genai_errors.APIError) and exc.code in (429, 503)

    else:
        print(f"(no usage counter for {type(llm).__name__}; token counts will read 0)")
        return

    state = {"last_call": 0.0}

    for name in methods:
        original = getattr(target, name)

        def wrapped(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
            import time

            elapsed = time.monotonic() - state["last_call"]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            attempt = 0
            while True:
                try:
                    response = _original(*args, **kwargs)
                    break
                except Exception as exc:  # noqa: BLE001 - re-raised if not transient
                    if not is_transient(exc) or attempt >= max_retries:
                        raise
                    delay = _gemini_retry_delay_seconds(exc)
                    reason = "rate limited" if exc.code == 429 else "server reports high demand"
                    print(f"    ({reason}; waiting {delay:.0f}s, "
                          f"attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    attempt += 1

            state["last_call"] = time.monotonic()
            usage.calls += 1
            i, o = read_usage(response)
            usage.input_tokens += i
            usage.output_tokens += o
            return response

        setattr(target, name, wrapped)


def main() -> int:
    from research_assistant.config_J import get_settings
    from research_assistant.llm_S import LLMError, get_llm, set_llm

    get_settings.cache_clear()
    settings = get_settings()

    key_by_backend = {
        "anthropic": ("anthropic_api_key", "RA_ANTHROPIC_API_KEY"),
        "gemini": ("gemini_api_key", "RA_GEMINI_API_KEY"),
    }
    if settings.judge_backend not in key_by_backend:
        print(
            f"RA_JUDGE_BACKEND is '{settings.judge_backend}', which has no real "
            f"model to smoke-test. Set it to 'gemini' or 'anthropic' in .env."
        )
        return 2
    attr, env_name = key_by_backend[settings.judge_backend]
    if not getattr(settings, attr):
        print(
            f"RA_JUDGE_BACKEND='{settings.judge_backend}' but {env_name} is unset. "
            f"Put it in the repo's .env file, then run this again."
        )
        return 2

    from research_assistant.agents.graph_S import gather, initial_state, run
    from research_assistant.judge.triage_S import assess_evidence
    from research_assistant.mcp_server.backend_S import set_backend
    from tests.fixtures.adversarial_corpus_S import (
        DISAGREE_A,
        DISAGREE_B,
        SUPERSEDED_NEW,
        SUPERSEDED_OLD,
        TOPICAL_NOT_ANSWERING,
        adversarial_backend,
    )

    set_llm(None)
    llm = get_llm()
    usage = Usage()
    instrument(llm, usage)
    model = settings.judge_model
    print(f"backend: {settings.judge_backend}  model: {model}\n")

    hard_failures = 0

    # --- 1-3: full pipeline on the built-in corpus -------------------------------
    print("== full pipeline, built-in corpus ==")
    for q in QUESTIONS:
        set_backend(None)
        try:
            state, handover = run(q)
        except LLMError as exc:
            print(f"  FAIL  {q[:50]}: {exc}")
            hard_failures += 1
            continue

        problems = []
        if state.draft is not None:
            for claim in state.draft.claims:
                if claim.text not in state.draft.text:
                    problems.append(f"claim not verbatim in draft: {claim.text[:60]!r}")
                if claim.chunk_id and claim.chunk_id not in state.evidence_ids:
                    problems.append(f"cites unretrieved chunk {claim.chunk_id}")
        route = " -> ".join(f"{s['route']}({s['trigger']})" for s in state.trajectory())
        status = "FAIL" if problems else "ok  "
        hard_failures += bool(problems)
        print(f"  {status}  {q[:50]}")
        print(f"        outcome={state.outcome}  claims="
              f"{len(state.draft.claims) if state.draft else 0}  route: {route}")
        for p in problems:
            print(f"        - {p}")
        if handover:
            print(f"        handover: {handover.trigger} — {handover.reason[:90]}")

    # --- 4: triage on the adversarial passages -----------------------------------
    print("\n== triage on adversarial passages (expected vs actual) ==")
    cases = [
        ("superseded result", (SUPERSEDED_OLD, SUPERSEDED_NEW),
         "Which fusion method works best for hybrid retrieval?", "conflicting", "freshness"),
        ("genuine disagreement", (DISAGREE_A, DISAGREE_B),
         "Is cross-encoder reranking necessary on scientific text?", "conflicting", "disagreement"),
        ("topical, does not answer", (TOPICAL_NOT_ANSWERING,),
         "Which fusion method works best for hybrid retrieval?", "insufficient", None),
    ]
    for name, chunks, question, want_label, want_kind in cases:
        set_backend(adversarial_backend(*chunks))
        state = gather(initial_state(question))
        a = state.assessment
        kinds = sorted({c.kind for c in a.conflicts}) if a else []
        label_ok = a is not None and a.label == want_label
        kind_ok = want_kind is None or want_kind in kinds
        mark = "match" if label_ok and kind_ok else "DIFF "
        print(f"  {mark}  {name}: expected {want_label}"
              f"{f'/{want_kind}' if want_kind else ''}, got "
              f"{a.label if a else None}{f'/{kinds}' if kinds else ''} "
              f"(confidence {a.confidence:.2f})" if a else "")
        if a and a.rationale:
            print(f"         rationale: {a.rationale[:110]}")

    # --- 5: run-to-run variance ----------------------------------------------------
    print("\n== run-to-run variance (same triage, twice) ==")
    set_backend(adversarial_backend(DISAGREE_A, DISAGREE_B))
    evidence = gather(initial_state("Is cross-encoder reranking necessary?")).evidence
    first = assess_evidence("Is cross-encoder reranking necessary?", evidence)
    second = assess_evidence("Is cross-encoder reranking necessary?", evidence)
    same = first.label == second.label
    print(f"  {'stable' if same else 'VARIED'}  {first.label} ({first.confidence:.2f}) vs "
          f"{second.label} ({second.confidence:.2f})")

    # --- summary -------------------------------------------------------------------
    set_backend(None)
    price_in, price_out = PRICES.get(model, (None, None))
    print(f"\n{usage.calls} calls, {usage.input_tokens:,} input / "
          f"{usage.output_tokens:,} output tokens", end="")
    if price_in is None:
        print("  (no price data for this model id)")
    elif price_in == 0.0 and price_out == 0.0:
        print("  ($0.00 — free tier)")
    else:
        cost = usage.input_tokens / 1e6 * price_in + usage.output_tokens / 1e6 * price_out
        print(f"  (about ${cost:.2f})")
    print("RESULT:", "hard checks passed" if hard_failures == 0
          else f"{hard_failures} hard check(s) failed")
    return 1 if hard_failures else 0


if __name__ == "__main__":
    sys.exit(main())
