"""Test-session hermeticity (Sude).

Every test in this suite is written assuming the deterministic stub backend,
either explicitly (`set_llm(ScriptedLLM(...))`) or implicitly (whatever
`get_llm()` returns by default). That default comes from `Settings`, which
reads `.env` — a file that legitimately exists now, carrying whichever real
backend and key a developer configured for the smoke script
(`scripts/smoke_real_model_S.py`) or their own local runs.

Without this fixture, a `.env` pointed at a real backend silently changes what
`pytest` does: tests that expect the stub's offline, extractive behaviour
instead make real network calls, with a placeholder or exhausted key, and fail
for a reason that has nothing to do with the code under test. That is exactly
the failure this caught — `RA_JUDGE_BACKEND=gemini` plus a placeholder key in
`.env` turned twenty passing tests into twenty 400 INVALID_ARGUMENT errors, none
of which pointed at an actual bug.

The fix is structural, not a patch for that one case: tests must not depend on
whichever backend a developer's machine happens to be configured for. This
fixture pins `RA_JUDGE_BACKEND=stub` for the whole session, session-scoped and
autouse so no test file needs to remember to ask for it, restoring the prior
environment value afterward for anything that inspects `os.environ` post-suite.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True, scope="session")
def _force_stub_judge_backend() -> Iterator[None]:
    import os

    from research_assistant.config_J import get_settings

    previous = os.environ.get("RA_JUDGE_BACKEND")
    os.environ["RA_JUDGE_BACKEND"] = "stub"
    get_settings.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("RA_JUDGE_BACKEND", None)
        else:
            os.environ["RA_JUDGE_BACKEND"] = previous
        get_settings.cache_clear()
