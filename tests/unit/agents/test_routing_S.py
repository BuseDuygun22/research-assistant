"""Routing policy tests (Sude).

Structured as an adversarial suite rather than a happy-path one: each test
injects a specific malformed or hostile verdict and asserts the router fails in
the *safe* direction. The property under test throughout is that no input
produces an unbounded loop or a silent misroute.
"""

from __future__ import annotations

import pytest

from research_assistant.agents.routing_S import Budget, decide_route
from research_assistant.agents.state_S import AgentState, ClaimSpan, Draft, StateViolation
from research_assistant.contracts.judge_J import (
    DEEP_VIOLATION_KINDS,
    SHALLOW_VIOLATION_KINDS,
    AnswerVerdict,
    FaithfulnessVerdict,
    FaithfulnessViolation,
    JudgeMeta,
    ViolationKind,
)

META = JudgeMeta(judge_model="stub", prompt_version="v1")


def fv(kinds=(), passed=False, confidence=1.0) -> FaithfulnessVerdict:
    return FaithfulnessVerdict(
        draft_id="d1",
        passed=passed,
        confidence=confidence,
        citation_precision=0.5,
        coverage=0.5,
        meta=META,
        violations=[
            FaithfulnessViolation(kind=k, claim=f"claim about {k}", explanation="x")
            for k in kinds
        ],
    )


def av(relevance=3, sufficient=True, confidence=1.0, missing=()) -> AnswerVerdict:
    return AnswerVerdict(
        draft_id="d1",
        query="q",
        relevance=relevance,
        evidence_sufficient=sufficient,
        missing_aspects=list(missing),
        confidence=confidence,
        meta=META,
    )


# --- the taxonomy is exhaustive ----------------------------------------------


def test_every_violation_kind_has_a_depth():
    """Principle 8, enforced at import: a kind added without a depth must not
    fall through to a default route."""
    from typing import get_args

    all_kinds = set(get_args(ViolationKind))
    classified = DEEP_VIOLATION_KINDS | SHALLOW_VIOLATION_KINDS
    assert all_kinds == classified
    assert not (DEEP_VIOLATION_KINDS & SHALLOW_VIOLATION_KINDS)


def test_unclassified_violation_escalates_rather_than_defaulting():
    """Belt and braces: even if the import guard were bypassed, an unrecognised
    kind must reach a human, not the writer."""
    v = fv(["missing_citation"])
    object.__setattr__(v.violations[0], "kind", "kind_from_the_future")
    d = decide_route(v, budget=Budget())
    assert d.route == "escalate"
    assert d.trigger == "unclassified_violation"


def test_failed_verdict_with_no_violations_escalates():
    """E2: a rubric that fails a draft without itemising why gives the writer
    nothing to fix. Unguided revision is the case that makes output worse, so
    escalate rather than hand over an empty repair brief."""
    d = decide_route(fv([], passed=False), budget=Budget())
    assert d.route == "escalate"
    assert d.trigger == "unexplained_failure"


# --- depth routing -----------------------------------------------------------


def test_shallow_routes_to_writer():
    d = decide_route(fv(["missing_citation"]), budget=Budget())
    assert (d.route, d.node) == ("rewrite", "writer")


def test_deep_routes_to_researcher():
    d = decide_route(fv(["unsupported_claim"]), budget=Budget())
    assert (d.route, d.node) == ("re_retrieve", "researcher")


def test_deep_dominates_mixed_violations():
    d = decide_route(fv(["missing_citation", "contradicts_source"]), budget=Budget())
    assert d.route == "re_retrieve"


# --- budgets are separate ----------------------------------------------------


def test_rewrite_budget_does_not_starve_re_retrieval():
    """The reason the counters are split: three rewrites must not consume the
    budget that a deep violation needs."""
    b = Budget(rewrites_used=3, max_rewrites=3, re_retrievals_used=0)
    d = decide_route(fv(["unsupported_claim"]), budget=b)
    assert d.route == "re_retrieve"


def test_exhausted_re_retrieval_escalates_instead_of_rewriting():
    """The dangerous fallback: with no retrieval budget left, a deep violation
    must NOT be handed to the writer, which could only delete or fabricate."""
    b = Budget(re_retrievals_used=2, max_re_retrievals=2)
    d = decide_route(fv(["unsupported_claim"]), budget=b)
    assert d.route == "escalate"
    assert d.trigger == "re_retrieval_budget_exhausted"


def test_exhausted_rewrite_budget_escalates():
    b = Budget(rewrites_used=3, max_rewrites=3)
    d = decide_route(fv(["missing_citation"]), budget=b)
    assert d.trigger == "rewrite_budget_exhausted"


def test_step_budget_is_the_global_backstop():
    b = Budget(steps_used=12, max_steps=12)
    d = decide_route(fv(["missing_citation"]), budget=b)
    assert d.trigger == "step_budget_exhausted"


def test_budget_spend_is_type_specific():
    b = Budget()
    assert b.spend("rewrite").rewrites_used == 1
    assert b.spend("rewrite").re_retrievals_used == 0
    assert b.spend("re_retrieve").re_retrievals_used == 1
    assert b.spend("escalate").steps_used == 1  # every route costs a step


def test_discover_spends_its_own_counter_not_re_retrievals():
    b = Budget(max_live_discoveries=1)
    spent = b.spend("discover")
    assert spent.live_discoveries_used == 1
    assert spent.re_retrievals_used == 0
    assert spent.live_discoveries_left == 0


def test_live_discovery_budget_is_zero_unless_configured():
    """Matches `graph_S.initial_state`: `max_live_discoveries` is only nonzero
    when `Settings.allow_live_discovery` is true."""
    assert Budget().max_live_discoveries == 0
    assert Budget().live_discoveries_left == 0


def test_discover_costs_more_than_a_local_re_retrieve():
    """Real network I/O and a full parse/chunk/embed pass are genuinely more
    expensive than reformulating and re-querying the local index - the cost
    model should make discovery the last resort, not a cheap first move."""
    from research_assistant.agents.routing_S import route_cost

    assert route_cost("discover") > route_cost("re_retrieve")


# --- uncertainty -------------------------------------------------------------


def test_uncertain_pass_escalates():
    d = decide_route(fv(passed=True, confidence=0.3), budget=Budget(),
                     min_faithfulness_confidence=0.5)
    assert d.trigger == "low_faithfulness_confidence"


def test_confident_failure_and_uncertain_failure_differ():
    """Principle 11: passed=False at .95 and at .51 are different events."""
    confident = decide_route(fv(["missing_citation"], confidence=0.95), budget=Budget(),
                             min_faithfulness_confidence=0.6)
    uncertain = decide_route(fv(["missing_citation"], confidence=0.51), budget=Budget(),
                             min_faithfulness_confidence=0.6)
    assert confident.route == "rewrite"
    assert uncertain.route == "escalate"


def test_uncalibrated_judge_does_not_escalate_everything():
    d = decide_route(fv(passed=True), budget=Budget(), min_faithfulness_confidence=0.0)
    assert d.route == "accept"


# --- the faithful-but-wrong cases --------------------------------------------


def test_faithful_but_does_not_answer_the_question():
    """The Goodhart case: perfectly grounded, answers nothing. Must not accept."""
    d = decide_route(fv(passed=True), budget=Budget(),
                     answer=av(relevance=0, missing=["the comparison"]))
    assert d.route == "re_retrieve"
    assert d.trigger == "answer_incomplete"


def test_faithful_and_partially_answers_is_accepted():
    """Grade 2 is good enough — demanding 3 would escalate every genuinely
    partial answer over a finite corpus."""
    d = decide_route(fv(passed=True), budget=Budget(), answer=av(relevance=2))
    assert d.route == "accept"


def test_insufficient_evidence_abstains_rather_than_retrying():
    """More retrieval over a corpus that lacks the answer produces a more
    confident wrong answer, not a right one."""
    d = decide_route(fv(passed=True), budget=Budget(), answer=av(sufficient=False))
    assert d.route == "abstain"
    assert d.node == "flag_for_human"


def test_abstention_outranks_incompleteness():
    d = decide_route(fv(passed=True), budget=Budget(),
                     answer=av(relevance=0, sufficient=False))
    assert d.route == "abstain"


def test_should_abstain_tries_live_discovery_before_conceding():
    """A full draft attempt still could not answer the question from the fixed
    corpus. With an unspent live-discovery budget, try expanding the corpus
    once before abstaining - this is the asymmetry the post-draft abstain path
    used to have relative to the pre-draft one (it never retried anything,
    budget or not): now both paths get one real chance to do something about
    it before giving up."""
    d = decide_route(fv(passed=True), budget=Budget(max_live_discoveries=1),
                     answer=av(sufficient=False))
    assert d.route == "discover"
    assert d.trigger == "live_discovery_attempt"


def test_should_abstain_still_abstains_once_live_discovery_is_spent():
    d = decide_route(
        fv(passed=True),
        budget=Budget(max_live_discoveries=1, live_discoveries_used=1),
        answer=av(sufficient=False),
    )
    assert d.route == "abstain"


def test_should_abstain_ignores_live_discovery_when_off_by_default():
    d = decide_route(fv(passed=True), budget=Budget(), answer=av(sufficient=False))
    assert d.route == "abstain"


def test_grounding_is_checked_before_relevance():
    """Judging the relevance of unsupported prose is meaningless."""
    d = decide_route(fv(["unsupported_claim"]), budget=Budget(), answer=av(relevance=0))
    assert d.trigger == "deep_violation"


def test_answer_verdict_is_optional():
    """The graph must still run before the answer judge exists."""
    d = decide_route(fv(passed=True), budget=Budget(), answer=None)
    assert d.route == "accept"


# --- termination -------------------------------------------------------------


@pytest.mark.parametrize("kinds", [["missing_citation"], ["unsupported_claim"], []])
def test_every_loop_terminates(kinds):
    """Principle 20/13: drive the router with an adversarial verdict that never
    passes and assert it stops."""
    state_budget = Budget()
    seen = []
    for _ in range(50):
        d = decide_route(fv(kinds, passed=False), budget=state_budget)
        seen.append(d.route)
        state_budget = state_budget.spend(d.route)
        if d.route in ("accept", "escalate", "abstain"):
            break
    assert seen[-1] in ("escalate", "abstain")
    assert len(seen) < 50, "router failed to terminate"


def test_every_decision_carries_a_reason():
    """Principle 31: an unexplained route is not debuggable."""
    for verdict in (fv(["missing_citation"]), fv(["unsupported_claim"]), fv(passed=True)):
        d = decide_route(verdict, budget=Budget())
        assert d.reason and len(d.reason) > 10
        assert d.trigger


# --- state -------------------------------------------------------------------


def test_node_cannot_write_a_field_it_does_not_own():
    s = AgentState(run_id="r1", question="q")
    with pytest.raises(StateViolation):
        s.apply("writer", evidence=())


def test_owner_can_write_its_own_field():
    s = AgentState(run_id="r1", question="q")
    assert s.apply("researcher", queries_issued=("q1",)).queries_issued == ("q1",)


def test_unknown_field_is_refused():
    s = AgentState(run_id="r1", question="q")
    with pytest.raises(StateViolation):
        s.apply("editor", nonexistent_field=1)


def test_history_is_preserved_across_revisions():
    s = AgentState(run_id="r1", question="q")
    s = s.add_draft(Draft(draft_id="d1", text="first"))
    s = s.add_draft(Draft(draft_id="d2", text="second", revision_of="d1"))
    assert len(s.drafts) == 2
    assert s.draft.draft_id == "d2"
    assert s.drafts[0].text == "first"


def test_repair_brief_keeps_verified_claims():
    """Principle 36/37: targeted correction, not wholesale regeneration."""
    claims = (
        ClaimSpan(claim_id="c1", text="good claim", chunk_id="x", verified=True),
        ClaimSpan(claim_id="c2", text="claim about missing_citation", verified=False),
    )
    s = AgentState(run_id="r1", question="q").add_draft(
        Draft(draft_id="d1", text="...", claims=claims)
    )
    s = s.apply("editor", faithfulness=fv(["missing_citation"]))
    brief = s.repair_brief()
    assert [c["claim_id"] for c in brief["claims_to_fix"]] == ["c2"]
    assert brief["keep"] == ["c1"]


def test_retrieval_brief_reports_what_was_already_tried():
    """So a reformulation explores rather than repeats."""
    s = AgentState(run_id="r1", question="q").apply("researcher", queries_issued=("first try",))
    s = s.apply("editor", faithfulness=fv(["unsupported_claim"]))
    brief = s.retrieval_brief()
    assert brief["already_tried"] == ["first try"]
    assert brief["unsupported_claims"] == ["claim about unsupported_claim"]


def test_recording_a_decision_charges_the_budget():
    s = AgentState(run_id="r1", question="q")
    d = decide_route(fv(["missing_citation"]), budget=s.budget)
    s = s.record(d)
    assert s.budget.rewrites_used == 1
    assert s.budget.steps_used == 1
    assert s.outcome == "pending"


def test_terminal_routes_set_the_outcome():
    s = AgentState(run_id="r1", question="q")
    s = s.record(decide_route(fv(passed=True), budget=s.budget))
    assert s.outcome == "accepted"
    assert s.is_terminal


def test_trajectory_is_readable():
    s = AgentState(run_id="r1", question="q")
    s = s.record(decide_route(fv(["missing_citation"]), budget=s.budget))
    s = s.record(decide_route(fv(passed=True), budget=s.budget))
    t = s.trajectory()
    assert [x["route"] for x in t] == ["rewrite", "accept"]
    assert t[0]["trigger"] == "shallow_violation"
