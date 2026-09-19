"""Tests for the eval gate's significance machinery (Sude).

The property that matters most is the negative one: a no-op change must come back
inconclusive. A gate that cannot say "I don't know" is a gate that promotes noise.
"""

from __future__ import annotations

import random

import pytest

from eval.metrics.significance_S import (
    AlignmentError,
    align,
    minimum_detectable_effect,
    observed_sd,
    paired_bootstrap,
    required_queries,
)


def noisy(n: int, mean: float, spread: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [min(1.0, max(0.0, rng.gauss(mean, spread))) for _ in range(n)]


# --- the negative property ---------------------------------------------------


def test_identical_runs_are_inconclusive():
    scores = noisy(120, 0.60, 0.18, seed=1)
    result = paired_bootstrap(scores, scores, metric="ndcg@5", seed=0)
    assert result.observed_diff == pytest.approx(0.0)
    assert result.significant is False
    assert result.direction == "inconclusive"


def test_pure_noise_is_usually_inconclusive():
    """Two runs drawn from the same distribution should not promote."""
    base = noisy(200, 0.60, 0.18, seed=2)
    cand = noisy(200, 0.60, 0.18, seed=3)
    result = paired_bootstrap(base, cand, metric="ndcg@5", seed=0)
    assert result.significant is False


# --- the positive property ---------------------------------------------------


def test_large_consistent_improvement_is_detected():
    base = noisy(150, 0.50, 0.12, seed=4)
    cand = [min(1.0, b + 0.08) for b in base]
    result = paired_bootstrap(base, cand, metric="ndcg@5", seed=0)
    assert result.significant is True
    assert result.direction == "improvement"
    assert result.ci_low > 0


def test_regression_is_detected_and_named():
    base = noisy(150, 0.60, 0.12, seed=5)
    cand = [max(0.0, b - 0.07) for b in base]
    result = paired_bootstrap(base, cand, seed=0)
    assert result.significant is True
    assert result.direction == "regression"
    assert result.ci_high < 0


def test_pairing_beats_not_pairing():
    """The reason the gate pairs at all: a consistent small shift is visible
    against high per-query variance only because difficulty cancels."""
    base = noisy(200, 0.55, 0.30, seed=6)
    cand = [min(1.0, b + 0.03) for b in base]
    paired = paired_bootstrap(base, cand, seed=0)
    # Shuffling the candidate destroys the pairing without changing either mean.
    shuffled = cand[:]
    random.Random(7).shuffle(shuffled)
    unpaired = paired_bootstrap(base, shuffled, seed=0)
    paired_width = paired.ci_high - paired.ci_low
    unpaired_width = unpaired.ci_high - unpaired.ci_low
    assert paired_width < unpaired_width
    assert paired.significant is True


# --- reproducibility ---------------------------------------------------------


def test_same_seed_gives_the_same_decision():
    base = noisy(80, 0.55, 0.20, seed=8)
    cand = noisy(80, 0.58, 0.20, seed=9)
    a = paired_bootstrap(base, cand, seed=42)
    b = paired_bootstrap(base, cand, seed=42)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


# --- strictness --------------------------------------------------------------


def test_mismatched_lengths_are_refused():
    with pytest.raises(AlignmentError):
        paired_bootstrap([0.1, 0.2, 0.3], [0.1, 0.2], seed=0)


def test_align_refuses_to_silently_intersect():
    """Dropping unmatched queries would shrink the eval set exactly when
    something upstream is broken."""
    with pytest.raises(AlignmentError) as exc:
        align({"q1": 0.5, "q2": 0.6}, {"q1": 0.5, "q3": 0.7})
    assert "different queries" in str(exc.value)


def test_align_orders_by_query_id():
    base, cand = align({"q2": 0.2, "q1": 0.1}, {"q2": 0.4, "q1": 0.3})
    assert base == [0.1, 0.2]
    assert cand == [0.3, 0.4]


def test_too_few_observations_is_an_error():
    with pytest.raises(ValueError):
        paired_bootstrap([0.5], [0.6], seed=0)


# --- sizing the eval set -----------------------------------------------------


def test_required_queries_matches_the_briefing_number():
    """The claim handed to Buse: ~200 queries to resolve a 3-5% difference at
    typical per-query spread. If this drifts, the labelling target drifts."""
    n = required_queries(target_effect=0.04, sd_of_differences=0.20)
    assert 150 <= n <= 250


def test_smaller_effects_need_more_queries():
    assert required_queries(0.02, 0.20) > required_queries(0.04, 0.20)


def test_required_queries_and_mde_are_inverses():
    sd = 0.20
    n = required_queries(target_effect=0.04, sd_of_differences=sd)
    assert minimum_detectable_effect(n, sd) == pytest.approx(0.04, abs=0.003)


def test_observed_sd_feeds_the_sizing_helpers():
    base = noisy(100, 0.55, 0.15, seed=10)
    cand = noisy(100, 0.58, 0.15, seed=11)
    sd = observed_sd(base, cand)
    assert sd > 0
    assert required_queries(0.05, sd) >= 2


def test_non_default_confidence_still_works():
    """Exercises the inverse-normal path, since scipy is not a dependency."""
    strict = required_queries(0.04, 0.20, confidence=0.99)
    normal = required_queries(0.04, 0.20, confidence=0.95)
    assert strict > normal
