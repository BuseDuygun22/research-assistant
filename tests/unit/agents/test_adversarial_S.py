"""Adversarial pipeline tests (Sude).

Principle 40: test against the cases built to break the design, not the ones that
confirm it. Each test runs the whole graph over a corpus fixture designed to trip
one failure, and asserts the system responds the way the design says it should.
"""

from __future__ import annotations

import pytest

from research_assistant.agents.graph_S import gather, initial_state, run
from research_assistant.agents.routing_S import Budget, decide_evidence_route
from research_assistant.contracts.judge_J import EvidenceAssessment, EvidenceConflict, JudgeMeta
from research_assistant.judge.triage_S import assess_evidence, describe_conflicts
from research_assistant.llm_S import set_llm
from research_assistant.mcp_server.backend_S import set_backend
from tests.fixtures.adversarial_corpus_S import (
    DISAGREE_A,
    DISAGREE_B,
    NEAR_DUPLICATE,
    PROMPT_INJECTION,
    SUPERSEDED_NEW,
    SUPERSEDED_OLD,
    TOPICAL_NOT_ANSWERING,
    AdversarialJudge,
    adversarial_backend,
)

META = JudgeMeta(judge_model="t", prompt_version="v1")


@pytest.fixture(autouse=True)
def reset():
    yield
    set_llm(None)
    set_backend(None)


def use(judge: AdversarialJudge, *chunks) -> AdversarialJudge:
    set_llm(judge)
    set_backend(adversarial_backend(*chunks))
    return judge


def assessment(label: str, conflicts=(), confidence: float = 0.9) -> EvidenceAssessment:
    return EvidenceAssessment(
        query="q", label=label, conflicts=list(conflicts), confidence=confidence, meta=META
    )


# --- E38: inter-document conflict ---------------------------------------------


def test_superseded_result_is_detected_as_freshness():
    use(AdversarialJudge(), SUPERSEDED_OLD, SUPERSEDED_NEW)
    state = gather(initial_state("which fusion method is best for hybrid retrieval"))
    assert state.assessment.label == "conflicting"
    [conflict] = state.assessment.conflicts
    assert conflict.kind == "freshness"
    assert conflict.newer_chunk_id == SUPERSEDED_NEW.chunk_id


def test_genuine_disagreement_is_detected():
    use(AdversarialJudge(), DISAGREE_A, DISAGREE_B)
    state = gather(initial_state("is cross-encoder reranking necessary"))
    assert [c.kind for c in state.assessment.conflicts] == ["disagreement"]


def test_conflict_reaches_the_writer_as_an_instruction():
    """Detection is worthless if the writer silently picks a side anyway. The
    conflict must travel into the prompt with an instruction not to resolve it."""
    judge = use(AdversarialJudge(), DISAGREE_A, DISAGREE_B)
    run("is cross-encoder reranking necessary for hybrid retrieval")
    writer_prompts = [u for name, u in judge.prompts if name == "_DraftOut"]
    assert writer_prompts, "writer never ran"
    assert "DISAGREEMENT" in writer_prompts[0]
    assert "Do not pick a side" in writer_prompts[0]


def test_superseded_instruction_says_to_lead_with_the_newer_result():
    a = assessment(
        "conflicting",
        [EvidenceConflict(kind="freshness", chunk_ids=["old", "new"],
                          summary="Reversed.", newer_chunk_id="new")],
    )
    text = describe_conflicts(a)
    assert "SUPERSEDED" in text and "Lead with new" in text


def test_conflicting_evidence_still_produces_a_draft():
    """A divided field is an answer, not a reason to go silent."""
    use(AdversarialJudge(), DISAGREE_A, DISAGREE_B)
    state, _ = run("is cross-encoder reranking necessary for hybrid retrieval")
    assert state.draft is not None
    assert state.trajectory()[0]["trigger"] == "evidence_conflicting"


def test_conflict_naming_a_passage_that_was_not_retrieved_is_dropped():
    """A conflict with a hallucinated id would route the run on evidence that
    does not exist."""

    class Liar(AdversarialJudge):
        def _triage(self, ids):
            return {"label": "conflicting", "confidence": 0.9, "conflicts": [
                {"kind": "disagreement", "chunk_ids": [ids[0], "0000deadbeef"],
                 "summary": "invented"}]}

    use(Liar(), DISAGREE_A)
    state = gather(initial_state("reranking"))
    assert state.assessment.conflicts == []


def test_conflicts_override_a_sufficient_label():
    """A judge that names conflicting passages and still says 'sufficient' has
    contradicted itself. The specific finding wins over the summary label."""

    class Inconsistent(AdversarialJudge):
        def _triage(self, ids):
            payload = super()._triage(ids)
            payload["label"] = "sufficient"
            return payload

    use(Inconsistent(), SUPERSEDED_OLD, SUPERSEDED_NEW)
    state = gather(initial_state("which fusion method is best"))
    assert state.assessment.label == "conflicting"


# --- E40: pre-generation triage ----------------------------------------------


def test_topical_but_useless_evidence_is_caught_before_the_writer_runs():
    """The confident-wrong failure: on-topic passages that do not answer. Caught
    at triage, so no writer call is spent on them."""
    judge = use(AdversarialJudge(), TOPICAL_NOT_ANSWERING)
    state, handover = run("which fusion method is best for hybrid retrieval")
    assert state.outcome == "abstained"
    assert state.drafts == ()
    assert not any(name == "_DraftOut" for name, _ in judge.prompts)
    assert handover is not None and handover.trigger == "evidence_insufficient"


def test_insufficient_evidence_retries_retrieval_before_abstaining():
    use(AdversarialJudge(), TOPICAL_NOT_ANSWERING)
    state, _ = run("which fusion method is best")
    routes = [d["route"] for d in state.trajectory()]
    assert routes.count("re_retrieve") == state.budget.max_re_retrievals
    assert routes[-1] == "abstain"


def test_empty_retrieval_is_insufficient_without_a_model_call():
    judge = AdversarialJudge()
    a = assess_evidence("anything", [], llm=judge)
    assert a.label == "insufficient"
    assert judge.prompts == []


def test_triage_failure_does_not_stop_the_run():
    """Triage is an optimisation. An optimisation that can halt the pipeline is
    a liability, so a failed triage degrades to the full draft-and-judge path."""

    class Broken(AdversarialJudge):
        def _triage(self, ids):
            raise ValueError("unparseable")

    use(Broken(), DISAGREE_A)
    state, _ = run("reranking")
    assert state.assessment.confidence == 0.0
    assert state.draft is not None


def test_low_confidence_triage_falls_through_rather_than_escalating():
    d = decide_evidence_route(assessment("insufficient", confidence=0.2),
                              budget=Budget(), min_confidence=0.5)
    assert d.route == "write"


def test_insufficient_with_no_budget_abstains_not_escalates():
    """A human given the same passages would reach the same conclusion."""
    d = decide_evidence_route(assessment("insufficient"),
                              budget=Budget(re_retrievals_used=2, max_re_retrievals=2))
    assert d.route == "abstain"


def test_partial_evidence_is_written_from():
    assert decide_evidence_route(assessment("partial"), budget=Budget()).route == "write"


# --- live discovery (opt-in corpus expansion, RA_ALLOW_LIVE_DISCOVERY) --------


def test_insufficient_with_local_budget_spent_tries_live_discovery_first():
    """Local re-retrieval exhausted, but a live-discovery attempt is still
    unspent: try expanding the corpus before conceding."""
    d = decide_evidence_route(
        assessment("insufficient"),
        budget=Budget(re_retrievals_used=2, max_re_retrievals=2, max_live_discoveries=1),
    )
    assert d.route == "discover"
    assert d.trigger == "live_discovery_attempt"
    assert d.node == "researcher"


def test_insufficient_still_abstains_once_live_discovery_is_also_spent():
    """Both budgets spent: this is the exact case
    `test_insufficient_with_no_budget_abstains_not_escalates` covers, now also
    proven with live discovery enabled but exhausted - the feature must not
    change the terminal outcome once there is truly nothing left to try."""
    d = decide_evidence_route(
        assessment("insufficient"),
        budget=Budget(
            re_retrievals_used=2,
            max_re_retrievals=2,
            live_discoveries_used=1,
            max_live_discoveries=1,
        ),
    )
    assert d.route == "abstain"
    assert d.trigger == "evidence_insufficient"


def test_live_discovery_is_off_by_default():
    """`Budget()`'s default `max_live_discoveries=0` - the feature must never
    activate for a caller who did not opt in, regardless of how exhausted the
    other budgets are."""
    d = decide_evidence_route(
        assessment("insufficient"), budget=Budget(re_retrievals_used=2, max_re_retrievals=2)
    )
    assert d.route == "abstain"


# --- robustness ----------------------------------------------------------------


def test_near_duplicates_do_not_inflate_the_evidence():
    """A preprint and its published version are one piece of evidence. Kept as
    two, they read as independent sources agreeing - corroboration that does
    not exist."""
    use(AdversarialJudge(), SUPERSEDED_NEW, NEAR_DUPLICATE)
    state = gather(initial_state("reciprocal rank fusion score blending"))
    assert len(state.evidence) == 1


def test_routing_does_not_depend_on_the_judge_resisting_injection():
    """A passage instructing the judge to pass everything. Even a judge that
    obeys it cannot change the *routing rules* — only the verdict the rules read.
    The guard that matters is that routing is code, not a model decision."""
    use(AdversarialJudge(obeys_injection=True), PROMPT_INJECTION)
    state, _ = run("what is hybrid retrieval")
    assert state.is_terminal
    for step in state.trajectory():
        assert step["trigger"]  # every decision still made by the rule set


def test_every_adversarial_run_terminates_within_budget():
    for chunks in (
        (SUPERSEDED_OLD, SUPERSEDED_NEW),
        (DISAGREE_A, DISAGREE_B),
        (TOPICAL_NOT_ANSWERING,),
        (PROMPT_INJECTION,),
        (SUPERSEDED_NEW, NEAR_DUPLICATE),
    ):
        use(AdversarialJudge(), *chunks)
        state, _ = run("hybrid retrieval fusion reranking")
        assert state.is_terminal
        assert state.budget.steps_used <= state.budget.max_steps
