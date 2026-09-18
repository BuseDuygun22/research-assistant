"""Generation-metric and gate-decision tests (Sude).

The theme is degenerate strategies: for each metric, what is the laziest system
that scores well on it, and does the metric suite as a whole reject that system?
"""

from __future__ import annotations

import pytest

from eval.metrics.generation_S import (
    answer_relevance,
    citation_coverage,
    citation_precision,
    context_recall,
    score_abstention,
    trajectory_metrics,
)
from eval.metrics.significance_S import (
    decide,
    holm_adjusted,
    paired_bootstrap,
    practically_significant,
)

# --- attribution metrics do not reward silence -------------------------------


def test_uncited_draft_scores_zero_not_one():
    """0/0 must not read as perfect. A vacuous-truth reading here would make
    'cite nothing' the optimal strategy against the gate."""
    assert citation_precision(verified=0, cited=0) == 0.0
    assert citation_coverage(cited=0, total_claims=0) == 0.0


def test_citation_precision_is_a_ratio_of_cited_claims():
    assert citation_precision(verified=3, cited=4) == 0.75


def test_quote_only_draft_maxes_attribution_but_not_relevance():
    """The Goodhart case, stated as a test: perfect attribution scores say
    nothing about whether the draft answered anything."""
    assert citation_precision(verified=5, cited=5) == 1.0
    assert answer_relevance([0, 0, 0]) == 0.0


# --- abstention is a classification problem ----------------------------------


def test_abstention_separates_the_four_cells():
    s = score_abstention([
        (True, True),    # correctly declined
        (True, False),   # answered an unanswerable question — the dangerous cell
        (False, True),   # declined an answerable one — timid
        (False, False),  # correctly answered
    ])
    assert s.correct_abstentions == 1
    assert s.missed_abstentions == 1
    assert s.abstention_recall == 0.5
    assert s.abstention_precision == 0.5


def test_always_abstaining_has_perfect_recall_and_poor_precision():
    """Why a single 'abstention rate' is uninterpretable."""
    s = score_abstention([(True, True), (False, True), (False, True)])
    assert s.abstention_recall == 1.0
    assert s.abstention_precision == pytest.approx(1 / 3)
    assert s.over_abstention_rate == 1.0


def test_never_abstaining_has_zero_recall():
    s = score_abstention([(True, False), (False, False)])
    assert s.abstention_recall == 0.0


# --- context recall catches cherry-picking -----------------------------------


def test_context_recall_detects_one_sided_retrieval():
    """Every claim can be supported while the picture is wrong, if retrieval
    surfaced half the relevant evidence."""
    assert context_recall(retrieved_ids=["a", "b"], relevant_ids=["a", "b", "c", "d"]) == 0.5


def test_context_recall_with_nothing_to_recall_is_not_a_failure():
    assert context_recall(retrieved_ids=["a"], relevant_ids=[]) == 1.0


# --- trajectory metrics ------------------------------------------------------


def test_trajectory_metrics_expose_the_path_not_just_the_answer():
    trajectories = [
        [{"route": "accept", "trigger": "passed"}],
        [
            {"route": "rewrite", "trigger": "shallow_violation"},
            {"route": "accept", "trigger": "passed"},
        ],
        [
            {"route": "re_retrieve", "trigger": "deep_violation"},
            {"route": "escalate", "trigger": "re_retrieval_budget_exhausted"},
        ],
    ]
    m = trajectory_metrics(trajectories)
    assert m["first_pass_accept_rate"] == pytest.approx(1 / 3)
    assert m["escalation_rate"] == pytest.approx(1 / 3)
    assert m["mean_re_retrievals"] == pytest.approx(1 / 3)
    assert m["trigger.deep_violation"] == pytest.approx(1 / 3)


def test_trajectory_metrics_on_no_runs_is_empty_not_a_crash():
    assert trajectory_metrics([]) == {}


# --- gate decisions ----------------------------------------------------------


def series(n, base, delta, spread, seed):
    import random

    rng = random.Random(seed)
    b = [min(1.0, max(0.0, rng.gauss(base, spread))) for _ in range(n)]
    return b, [min(1.0, max(0.0, x + delta)) for x in b]


def test_statistically_significant_but_trivially_small_is_rejected():
    """Principle 29: with enough queries a +0.002 change becomes detectable and
    still does not matter."""
    b, c = series(400, 0.60, 0.002, 0.10, seed=1)
    r = paired_bootstrap(b, c, metric="ndcg@5", seed=0)
    assert r.significant is True
    assert practically_significant(r, min_effect=0.02) is False
    assert decide([r], min_effect=0.02).promote is False


def test_large_improvement_promotes():
    b, c = series(200, 0.55, 0.06, 0.12, seed=2)
    r = paired_bootstrap(b, c, metric="ndcg@5", seed=0)
    v = decide([r], min_effect=0.02)
    assert v.promote is True
    assert "ndcg@5" in v.reason


def test_a_regression_blocks_promotion_even_beside_a_win():
    """Asymmetric on purpose: a measurable regression is worth looking at even
    when a different metric improved more."""
    b1, c1 = series(200, 0.55, 0.06, 0.12, seed=3)
    b2, c2 = series(200, 0.70, -0.05, 0.12, seed=4)
    good = paired_bootstrap(b1, c1, metric="ndcg@5", seed=0)
    bad = paired_bootstrap(b2, c2, metric="citation_precision", seed=0)
    v = decide([good, bad], min_effect=0.02)
    assert v.promote is False
    assert "citation_precision" in v.reason


def test_no_metrics_does_not_promote():
    assert decide([]).promote is False


def test_holm_is_stricter_than_testing_each_metric_alone():
    """Principle 28: four metrics at 5% each is not a 5% false-positive rate."""
    noise = [paired_bootstrap(*series(120, 0.6, 0.0, 0.15, seed=s), seed=0) for s in range(4)]
    assert sum(holm_adjusted(noise)) <= sum(r.significant for r in noise)


def test_gate_report_is_readable():
    b, c = series(150, 0.55, 0.06, 0.12, seed=9)
    v = decide([paired_bootstrap(b, c, metric="ndcg@5", seed=0)], min_effect=0.02)
    text = v.report()
    assert "PROMOTE" in text
    assert "ndcg@5" in text


def test_holm_does_not_reject_a_real_improvement_across_metrics():
    """Regression: an earlier Holm implementation ranked on the CI bound and then
    compared the *confidence level* against the corrected threshold. Because a 95%
    interval reports 0.95 for every metric regardless of evidence, that rejected
    everything whenever more than one metric was tested — a correction that turned
    every multi-metric run inconclusive. Correction needs a quantity that varies
    with the evidence, so it operates on the bootstrap p-value."""
    b, c = series(120, 0.60, 0.05, 0.15, seed=21)
    results = [
        paired_bootstrap(b, c, metric="ndcg@5", seed=0),
        paired_bootstrap(b, c, metric="coverage", seed=0),
    ]
    assert all(holm_adjusted(results)), "Holm rejected a clear two-metric improvement"
    assert decide(results, min_effect=0.02).promote is True


def test_holm_still_rejects_pure_noise():
    noise = [
        paired_bootstrap(*series(120, 0.6, 0.0, 0.15, seed=s), metric=f"m{s}", seed=0)
        for s in range(4)
    ]
    assert not any(holm_adjusted(noise))


def test_p_value_is_floored_at_bootstrap_resolution():
    """The bootstrap cannot resolve a p smaller than 1/n_resamples; reporting 0.0
    would claim more than it measured."""
    b, c = series(100, 0.5, 0.20, 0.05, seed=22)
    r = paired_bootstrap(b, c, seed=0, n_resamples=500)
    assert r.p_value >= 1 / 500
