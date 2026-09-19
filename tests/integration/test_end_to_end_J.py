"""End-to-end integration (JOINT).

Runs the whole pipeline — retrieval, writing, judging, routing — against the stub
backend and the stub LLM. It asserts the *seams* hold, not that the output is
good: the stub cannot produce good output and is not supposed to.

The value is regression coverage on the joins. Unit tests pass while the pieces
do not fit together; this fails when they do not.
"""

from __future__ import annotations

import pytest

from research_assistant.agents.graph_S import run
from research_assistant.agents.routing_S import NON_TERMINAL_TRIGGERS
from research_assistant.llm_S import set_llm
from research_assistant.mcp_server.backend_S import set_backend

QUESTIONS = [
    "How does reciprocal rank fusion compare to weighted score blending?",
    "What does Lost in the Middle say about context position?",
    "Why use DPO instead of a reward model and PPO?",
]


@pytest.fixture(autouse=True)
def reset():
    yield
    set_llm(None)
    set_backend(None)


@pytest.mark.parametrize("question", QUESTIONS)
def test_pipeline_reaches_a_terminal_state(question):
    state, _ = run(question)
    assert state.is_terminal
    assert state.outcome in ("accepted", "abstained", "escalated")


@pytest.mark.parametrize("question", QUESTIONS)
def test_pipeline_stays_within_budget(question):
    """Every loop terminates by a budget, not by luck."""
    state, _ = run(question)
    assert state.budget.rewrites_used <= state.budget.max_rewrites
    assert state.budget.re_retrievals_used <= state.budget.max_re_retrievals
    assert state.budget.steps_used <= state.budget.max_steps


def test_every_decision_is_explainable():
    """Principle: an unexplained route is not debuggable. Holds end to end, not
    just in the router's unit tests."""
    state, _ = run(QUESTIONS[0])
    for step in state.trajectory():
        assert step["trigger"]
        assert step["reason"]


def test_terminal_trigger_is_actually_terminal():
    state, _ = run(QUESTIONS[0])
    last = state.trajectory()[-1]
    assert last["trigger"] not in NON_TERMINAL_TRIGGERS or last["route"] in (
        "accept",
        "abstain",
        "escalate",
    )


def test_claims_trace_to_retrieved_evidence():
    """The seam that matters most: a cited chunk_id must be one retrieval
    actually returned, or the citation is unverifiable."""
    state, _ = run(QUESTIONS[0])
    assert state.draft is not None
    for claim in state.draft.claims:
        if claim.chunk_id is not None:
            assert claim.chunk_id in state.evidence_ids


def test_state_write_ownership_holds_through_a_real_run():
    state, _ = run(QUESTIONS[0])
    # The run completed without a StateViolation, and the fields each node owns
    # are populated by the end of it.
    assert state.evidence
    assert state.drafts
    assert state.decisions


def test_escalated_runs_produce_an_actionable_handover():
    """If any question escalates under the stub, the handover must be usable."""
    for question in QUESTIONS:
        state, handover = run(question)
        if handover is None:
            continue
        rendered = handover.render()
        assert handover.next_step
        assert "trajectory" in rendered
        assert handover.trigger in rendered
