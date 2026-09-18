"""Calibration, cost, and granularity tests (Sude).

Calibration runs on synthetic labels here because Buse's qrels do not exist yet.
The point is that the instrument is correct before the data arrives, so the first
real run produces a number worth believing rather than a debugging session.
"""

from __future__ import annotations

import random

import pytest

from eval.metrics.generation_S import claim_granularity, granularity_sweep
from research_assistant.agents.routing_S import (
    HUMAN_COST,
    Budget,
    decide_route,
    route_cost,
)
from research_assistant.contracts.judge_J import (
    FaithfulnessVerdict,
    FaithfulnessViolation,
    JudgeMeta,
)
from research_assistant.judge.calibration_S import (
    TemperatureScaler,
    cohens_kappa,
    krippendorff_alpha_nominal,
    reliability,
    suggest_escalation_threshold,
)

# --- agreement -----------------------------------------------------------------


def test_perfect_agreement_is_kappa_one():
    r = cohens_kappa([0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 0, 1])
    assert r.kappa == pytest.approx(1.0)


def test_chance_agreement_is_near_zero():
    rng = random.Random(1)
    human = [rng.randrange(4) for _ in range(2000)]
    judge = [rng.randrange(4) for _ in range(2000)]
    assert abs(cohens_kappa(human, judge).kappa) < 0.05


def test_constant_judge_on_skewed_labels_is_flagged():
    """The trap: 90% grade 0, a judge that always says 0 gets 90% observed
    agreement and a kappa of 0 — and neither number alone says it learned
    nothing. The report must say it."""
    human = [0] * 90 + [3] * 10
    judge = [0] * 100
    r = cohens_kappa(human, judge)
    assert r.observed_agreement == pytest.approx(0.9)
    assert r.kappa == pytest.approx(0.0)
    assert r.is_degenerate
    assert "WARNING" in r.render()


def test_report_exposes_where_the_disagreements_are():
    r = cohens_kappa([2, 2, 2, 3, 3], [3, 3, 2, 3, 3])
    assert r.biggest_disagreements()[0] == (2, 3, 2)


def test_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cohens_kappa([0, 1], [0])


def test_alpha_handles_three_annotators_with_gaps():
    """Principle 23: once qrels are double-labelled, agreement across annotators
    with partial coverage needs alpha, not kappa."""
    ratings = [[1, 1, None], [2, 2, 2], [0, None, 0], [3, 3, 3]]
    assert krippendorff_alpha_nominal(ratings) == pytest.approx(1.0)


def test_alpha_needs_at_least_one_doubly_rated_item():
    with pytest.raises(ValueError):
        krippendorff_alpha_nominal([[1, None], [None, 2]])


# --- calibration -----------------------------------------------------------------


def overconfident(n: int, seed: int) -> tuple[list[float], list[bool]]:
    """A judge that says ~0.95 and is right ~70% of the time."""
    rng = random.Random(seed)
    confs = [min(0.999, max(0.5, rng.gauss(0.95, 0.03))) for _ in range(n)]
    correct = [rng.random() < 0.7 for _ in range(n)]
    return confs, correct


def test_temperature_scaling_damps_an_overconfident_judge():
    confs, correct = overconfident(500, seed=2)
    scaler = TemperatureScaler().fit(confs, correct)
    assert scaler.temperature > 1.0
    before = reliability(confs, correct).expected_calibration_error
    after = reliability([scaler.apply(c) for c in confs], correct).expected_calibration_error
    assert after < before


def test_unfitted_scaler_is_identity():
    assert TemperatureScaler().apply(0.8) == pytest.approx(0.8)


def test_reliability_reports_direction_not_just_error():
    confs, correct = overconfident(300, seed=3)
    report = reliability(confs, correct)
    top = [b for b in report.bins if b.count and b.lower >= 0.9]
    assert top and top[0].gap > 0  # overconfident in exactly the range a threshold would sit
    assert "over" in report.render()


def test_threshold_is_derived_when_the_judge_is_good_enough():
    confs = [0.3] * 50 + [0.9] * 50
    correct = [False] * 50 + [True] * 50
    assert suggest_escalation_threshold(confs, correct, target_precision=0.95) == 0.9


def test_no_threshold_is_a_real_answer():
    """Returning 1.0 would escalate everything and look like a setting rather
    than a failure. None says the judge is not good enough to gate on yet."""
    confs, correct = overconfident(200, seed=4)
    assert suggest_escalation_threshold(confs, correct, target_precision=0.99) is None


# --- cost ------------------------------------------------------------------------


def test_re_retrieval_costs_more_than_a_rewrite():
    assert route_cost("re_retrieve") > route_cost("rewrite") > route_cost("accept")


def test_budget_accumulates_cost_by_route():
    b = Budget().spend("rewrite").spend("re_retrieve")
    assert b.cost_spent == pytest.approx(route_cost("rewrite") + route_cost("re_retrieve"))


def test_escalating_early_only_when_finishing_costs_more_than_a_human():
    assert Budget().should_escalate_early() is False
    expensive = Budget(cost_spent=HUMAN_COST)
    assert expensive.should_escalate_early() is True


def test_cost_ceiling_is_off_unless_asked_for():
    """It trades autonomy for spend; an operator decides that, not a default."""
    v = FaithfulnessVerdict(
        draft_id="d", passed=False, citation_precision=1.0, coverage=1.0,
        meta=JudgeMeta(judge_model="t", prompt_version="v1"),
        violations=[FaithfulnessViolation(kind="missing_citation", claim="c", explanation="e")],
    )
    spent = Budget(cost_spent=HUMAN_COST)
    assert decide_route(v, budget=spent).route == "rewrite"
    d = decide_route(v, budget=spent, cost_aware=True)
    assert (d.route, d.trigger) == ("escalate", "cost_ceiling")


# --- granularity (E39) -----------------------------------------------------------


def test_granularity_profiles_a_decomposition():
    draft = "RRF beats blending. It needs no tuning."
    fine = claim_granularity(draft, ["RRF beats", "blending loses", "no tuning", "it"])
    coarse = claim_granularity(draft, ["RRF beats blending.", "It needs no tuning."])
    assert fine.claims_per_sentence > coarse.claims_per_sentence
    assert fine.mean_words_per_claim < coarse.mean_words_per_claim


def test_sweep_picks_the_best_level():
    best, means = granularity_sweep(
        {"atomic": [0.80, 0.82], "clause": [0.91, 0.89], "sentence": [0.85, 0.84]}
    )
    assert best == "clause"
    assert means["clause"] == pytest.approx(0.90)


def test_sweep_refuses_levels_scored_on_different_drafts():
    with pytest.raises(ValueError):
        granularity_sweep({"atomic": [0.8, 0.9], "sentence": [0.7]})
