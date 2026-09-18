"""Regression: the test session must not depend on a developer's .env (JOINT).

The concrete failure this guards: `.env` set `RA_JUDGE_BACKEND=gemini` with a
placeholder key, and twenty otherwise-passing tests turned into real 400
INVALID_ARGUMENT errors — a config problem on the developer's machine, not a
bug in the code under test. `tests/conftest.py` pins the backend to `stub` for
the whole session; this test is what would have caught that regression before
it needed to be diagnosed from a stack trace.
"""

from __future__ import annotations

from research_assistant.config_J import get_settings


def test_default_backend_is_stub_regardless_of_env_file():
    """Whatever a local .env sets, the test session runs on the stub."""
    assert get_settings().judge_backend == "stub"


def test_get_llm_returns_the_stub_without_any_key_configured():
    """The property that actually protects every other test: constructing the
    default client here must never attempt a network call."""
    from research_assistant.llm_S import StubLLM, get_llm, set_llm

    set_llm(None)
    assert isinstance(get_llm(), StubLLM)
