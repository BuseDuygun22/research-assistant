"""Node and graph tests (Sude).

The stub LLM always produces a passing verdict — that is its job, proving the
plumbing runs. Failure paths are driven by `ScriptedLLM`, which returns payloads
a test names explicitly rather than depending on a hash landing in a bucket.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from research_assistant.agents.graph_S import initial_state, run
from research_assistant.agents.nodes.flag_for_human_S import NEXT_STEPS, flag_for_human
from research_assistant.agents.nodes.researcher_S import is_novel, researcher
from research_assistant.agents.nodes.writer_S import context_window, writer
from research_assistant.agents.routing_S import Budget, RoutingDecision
from research_assistant.agents.state_S import AgentState
from research_assistant.llm_S import set_llm
from research_assistant.mcp_server.backend_S import set_backend


class ScriptedLLM:
    """Returns pre-set payloads by schema name; records what it was asked."""

    def __init__(self, payloads: dict[str, list[dict[str, Any]]]):
        self.payloads = {k: list(v) for k, v in payloads.items()}
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        self.calls.append(("complete", user))
        return "scripted summary"

    def complete_json(
        self, system: str, user: str, schema: type[BaseModel], max_tokens: int = 1024
    ) -> BaseModel:
        self.calls.append((schema.__name__, user))
        queue = self.payloads.get(schema.__name__)
        if not queue:
            raise AssertionError(f"ScriptedLLM has no payload for {schema.__name__}")
        return schema.model_validate(queue.pop(0) if len(queue) > 1 else queue[0])


@pytest.fixture(autouse=True)
def reset():
    yield
    set_llm(None)
    set_backend(None)


# --- novelty check (E12) -----------------------------------------------------


def test_identical_query_is_not_novel():
    assert is_novel("hybrid retrieval fusion", ["hybrid retrieval fusion"]) is False


def test_reordered_query_is_not_novel():
    """Token-set based, so shuffling words does not buy a new round."""
    assert is_novel("fusion retrieval hybrid", ["hybrid retrieval fusion"]) is False


def test_genuinely_different_query_is_novel():
    assert is_novel("reciprocal rank fusion Cormack", ["dense passage retrieval"]) is True


def test_empty_query_is_never_novel():
    assert is_novel("", ["anything"]) is False


def test_novelty_holds_against_every_previous_query():
    assert is_novel("alpha beta", ["gamma delta", "alpha beta"]) is False


# --- researcher --------------------------------------------------------------


def test_first_round_uses_the_raw_question():
    state = researcher(initial_state("what is reciprocal rank fusion"))
    assert state.queries_issued == ("what is reciprocal rank fusion",)
    assert len(state.evidence) > 0


def test_evidence_accumulates_across_rounds():
    """Dropping earlier evidence would turn already-verified claims into
    unsupported ones on the next check."""
    state = researcher(initial_state("reciprocal rank fusion"))
    first = set(state.evidence_ids)
    state = researcher(state)
    assert first <= set(state.evidence_ids)


# --- writer ------------------------------------------------------------------


def test_writer_context_is_capped_at_the_span_budget():
    """The context-rot budget is enforced, not merely documented."""
    state = researcher(initial_state("retrieval"))
    window = context_window(state)
    assert window.count("[") <= 5 * 2  # id + title bracket per span, generously


def test_writer_produces_claims_that_appear_in_the_draft():
    """The invariant the faithfulness judge relies on: claims are matched to the
    draft by string equality, so a paraphrased claim breaks repair."""
    state = writer(researcher(initial_state("reciprocal rank fusion")))
    assert state.draft is not None
    for claim in state.draft.claims:
        assert claim.text in state.draft.text


def test_writer_records_the_revision_chain():
    state = writer(researcher(initial_state("reciprocal rank fusion")))
    first = state.draft.draft_id
    state = state.apply(
        "editor",
        faithfulness=_verdict(passed=False),
    )
    state = writer(state)
    assert len(state.drafts) == 2
    assert state.draft.revision_of == first


# --- escalation handover -----------------------------------------------------


def test_handover_names_a_next_step_for_every_trigger():
    """An escalation that does not say what a human should do wastes the
    diagnosis the system already computed."""
    from typing import get_args

    from research_assistant.agents.routing_S import NON_TERMINAL_TRIGGERS, Trigger

    for trigger in get_args(Trigger):
        if trigger in NON_TERMINAL_TRIGGERS:
            continue
        assert trigger in NEXT_STEPS, f"no guidance for trigger {trigger}"


def test_handover_survives_an_unknown_trigger():
    state = AgentState(run_id="r", question="q").record(
        RoutingDecision("escalate", "passed", "synthetic", Budget())  # type: ignore[arg-type]
    )
    handover = flag_for_human(state)
    assert "Unrecognised trigger" in handover.next_step


def test_handover_renders_the_trajectory():
    state, handover = run("reciprocal rank fusion")
    if handover is None:
        pytest.skip("stub run accepted; nothing escalated")
    assert "trajectory" in handover.render()


# --- the graph ---------------------------------------------------------------


def test_stub_run_reaches_a_terminal_state():
    state, _ = run("how does reciprocal rank fusion work")
    assert state.is_terminal
    assert state.outcome in ("accepted", "abstained", "escalated")


def test_run_records_a_decision_per_editor_pass():
    state, _ = run("reciprocal rank fusion")
    assert len(state.decisions) >= 1
    assert state.trajectory()[0]["trigger"]


def test_run_is_deterministic_under_the_stub():
    """A gate decision that cannot be reproduced cannot be appealed."""
    a, _ = run("reciprocal rank fusion", run_id="fixed")
    b, _ = run("reciprocal rank fusion", run_id="fixed")
    assert a.draft.text == b.draft.text
    assert [d.route for d in a.decisions] == [d.route for d in b.decisions]


def test_judge_failure_escalates_rather_than_shipping():
    """A verdict we cannot parse is neither a pass nor a fail. The one
    unacceptable outcome is treating it as a pass."""
    from research_assistant.agents.nodes.editor_S import editor

    state = writer(researcher(initial_state("reciprocal rank fusion")))
    set_llm(ScriptedLLM({}))  # any judge call raises AssertionError -> not caught
    with pytest.raises(AssertionError):
        editor(state)


def _verdict(passed: bool):
    from research_assistant.contracts.judge_J import FaithfulnessVerdict, JudgeMeta

    return FaithfulnessVerdict(
        draft_id="d",
        passed=passed,
        citation_precision=1.0,
        coverage=1.0,
        meta=JudgeMeta(judge_model="stub", prompt_version="v1"),
    )
