"""Contract tests (JOINT).

These pin the behaviour both tracks build against. A change here is a change to
the seam, so a failure means "renegotiate", not "fix the test".
"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from research_assistant.contracts.judge_J import (
    DEEP_VIOLATION_KINDS,
    SHALLOW_VIOLATION_KINDS,
    AnswerVerdict,
    FaithfulnessVerdict,
    FaithfulnessViolation,
    JudgeMeta,
    ViolationKind,
)
from research_assistant.contracts.mcp_tools_J import ToolError
from research_assistant.contracts.retrieval_J import Chunk

META = JudgeMeta(judge_model="stub", prompt_version="v1")


def violation(kind: str) -> FaithfulnessViolation:
    return FaithfulnessViolation(
        kind=kind,  # type: ignore[arg-type]
        claim="RRF outperforms weighted score blending.",
        cited_chunk_ids=["c1"],
        explanation="test fixture",
    )


def verdict(**kw) -> FaithfulnessVerdict:
    base = dict(
        draft_id="d1",
        violations=[],
        citation_precision=1.0,
        coverage=1.0,
        passed=True,
        meta=META,
    )
    base.update(kw)
    return FaithfulnessVerdict(**base)  # type: ignore[arg-type]


# --- chunk id determinism ----------------------------------------------------


def test_derive_id_is_deterministic():
    """The rule Buse's qrels depend on: same text, same id, forever."""
    a = Chunk.derive_id("2004.04906", 0, 120, "Dense passage retrieval...")
    b = Chunk.derive_id("2004.04906", 0, 120, "Dense passage retrieval...")
    assert a == b


def test_derive_id_changes_when_chunking_changes():
    """A changed boundary must be a loud failure, not a silent relabel."""
    a = Chunk.derive_id("2004.04906", 0, 120, "Dense passage retrieval...")
    b = Chunk.derive_id("2004.04906", 0, 140, "Dense passage retrieval...")
    assert a != b


# --- violation depth ---------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    ["unsupported_claim", "inferential_leap", "contradicts_source"],
)
def test_deep_kinds_need_new_evidence(kind):
    assert violation(kind).is_deep is True
    assert kind in DEEP_VIOLATION_KINDS


@pytest.mark.parametrize(
    "kind",
    ["missing_citation", "overstated_certainty", "misattributed_citation"],
)
def test_shallow_kinds_are_fixable_by_rewriting(kind):
    assert violation(kind).is_deep is False


def test_inferential_leap_is_an_accepted_kind():
    """Added because 2026 evaluations attribute most residual citation error to
    inferential linking rather than to wholly uncited claims."""
    assert violation("inferential_leap").kind == "inferential_leap"


def test_unknown_kind_is_rejected():
    with pytest.raises(ValidationError):
        violation("vibes_based_objection")


# --- violation partitioning ---------------------------------------------------
#
# Routing itself is tested in tests/unit/agents/test_routing_S.py — it needs
# budget state, which is orchestration rather than contract. What the contract
# owes the router is a taxonomy with no gaps.


def test_violations_partition_by_depth():
    v = verdict(
        passed=False,
        violations=[
            violation("missing_citation"),
            violation("overstated_certainty"),
            violation("contradicts_source"),
        ],
    )
    assert len(v.shallow_violations) == 2
    assert len(v.deep_violations) == 1
    assert v.unclassified_violations == []


def test_depth_sets_are_exhaustive_and_disjoint():
    """Principle 8 at the type level: a kind that belongs to neither set would
    fall through to the router's default branch, so the module refuses to import
    in that state."""
    all_kinds = set(get_args(ViolationKind))
    assert all_kinds == DEEP_VIOLATION_KINDS | SHALLOW_VIOLATION_KINDS
    assert not (DEEP_VIOLATION_KINDS & SHALLOW_VIOLATION_KINDS)


def test_confidence_defaults_to_trusting_the_verdict():
    """An uncalibrated judge must degrade to always-trust, not escalate
    everything."""
    assert verdict(passed=True).confidence == 1.0


def test_confidence_is_bounded():
    with pytest.raises(ValidationError):
        verdict(confidence=1.4)


# --- answer verdict -----------------------------------------------------------


def answer(relevance=3, sufficient=True) -> AnswerVerdict:
    return AnswerVerdict(
        draft_id="d1",
        query="q",
        relevance=relevance,
        evidence_sufficient=sufficient,
        meta=META,
    )


def test_partial_answers_count_as_answering():
    """Grade 2 passes: demanding 3 would escalate every genuinely partial answer
    over a finite corpus."""
    assert answer(relevance=2).answers_question is True
    assert answer(relevance=1).answers_question is False


def test_abstention_is_distinct_from_not_answering():
    """A draft can fail to answer a question the corpus *could* have answered —
    that is a retrieval miss, not grounds to abstain."""
    missed = answer(relevance=0, sufficient=True)
    assert missed.answers_question is False
    assert missed.should_abstain is False

    unanswerable = answer(relevance=0, sufficient=False)
    assert unanswerable.should_abstain is True


def test_faithfulness_and_answer_relevance_are_independent():
    """The degenerate case the AnswerVerdict exists to catch: five verbatim
    quotes score perfectly on attribution and answer nothing."""
    quotes_only = verdict(passed=True, citation_precision=1.0, coverage=1.0)
    assert quotes_only.passed is True
    assert answer(relevance=0).answers_question is False


# --- tool errors -------------------------------------------------------------


def test_tool_error_defaults_are_conservative():
    err = ToolError(code="chunk_not_found", message="unknown id")
    assert err.retryable is False
    assert err.partial is False
    assert err.remediation is None


def test_tool_error_carries_remediation_and_partial():
    err = ToolError(
        code="no_results",
        message="No chunks matched.",
        retryable=True,
        remediation="Broaden the query or drop the year filter.",
        partial=False,
    )
    assert err.remediation.startswith("Broaden")
    assert err.retryable is True


def test_tool_error_forbids_unknown_fields():
    """extra='forbid' is what keeps the seam from drifting silently."""
    with pytest.raises(ValidationError):
        ToolError(code="x", message="y", retrayable=True)  # type: ignore[call-arg]
