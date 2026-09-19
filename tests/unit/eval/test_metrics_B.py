"""Tests for `eval.metrics.retrieval_B` (Track A, Buse).

These metrics decide what ships, so a wrong metric is the worst bug in the repo: it
does not crash, it just makes every promotion decision wrong in the same direction and
nothing downstream can tell. Every expected value below is therefore **hand-computed
and written as a literal**, not produced by re-implementing the formula in the test
(which would only prove the code agrees with itself).

Gain convention under test: exponential gain, ``(2**grade - 1) / log2(rank + 1)``.

    grade 3 -> gain 7      rank 1 -> discount 1 / log2(2) = 1
    grade 2 -> gain 3      rank 2 -> discount 1 / log2(3) = 0.6309297535714574
    grade 1 -> gain 1      rank 3 -> discount 1 / log2(4) = 0.5
    grade 0 -> gain 0      rank 4 -> discount 1 / log2(5) = 0.4306765580733931

Fixtures are in-memory. There is no corpus and no real held-out set yet, and these
tests must never depend on one: they test arithmetic, not the retriever.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eval.metrics.retrieval_B import (
    CompareVerdict,
    EvalSet,
    EvalSetError,
    citation_precision,
    compare,
    evaluate,
    failure_breakdown,
    load_eval_set,
    metric_names,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

# A query whose labels span the whole grade scale.
REL = {"a": 3, "b": 2, "c": 1, "d": 0}

# Ideal DCG@3 for REL = 7/1 + 3/log2(3) + 1/log2(4)
#                     = 7 + 1.8927892607143721 + 0.5
#                     = 9.392789260714372
IDEAL_DCG3 = 9.392789260714372


# ======================================================================================
# nDCG
# ======================================================================================


def test_ndcg_perfect_ranking_is_exactly_one() -> None:
    """The ideal ordering scores 1.0. If this drifts, every other number is meaningless."""
    assert ndcg_at_k(["a", "b", "c"], REL, k=3) == pytest.approx(1.0)


def test_ndcg_perfect_ranking_with_distractors_below_is_still_one() -> None:
    """Grade-0 chunks *after* the relevant ones add no gain and must not lower the score."""
    assert ndcg_at_k(["a", "b", "c", "d", "zzz"], REL, k=5) == pytest.approx(1.0)


def test_ndcg_reversed_ranking_scores_lower() -> None:
    """Exact reversal of the ideal order.

    DCG([1, 2, 3]) = 1/1 + 3/log2(3) + 7/log2(4)
                   = 1 + 1.8927892607143721 + 3.5
                   = 6.392789260714372
    nDCG = 6.392789260714372 / 9.392789260714372 = 0.6806060567602009
    """
    reversed_score = ndcg_at_k(["c", "b", "a"], REL, k=3)
    assert reversed_score == pytest.approx(0.6806060567602009, abs=1e-12)
    assert reversed_score < ndcg_at_k(["a", "b", "c"], REL, k=3)


def test_ndcg_is_position_sensitive() -> None:
    """One grade-3 chunk, moved down the list. Score must be strictly monotone in rank.

    With a single relevant chunk the ideal DCG is 7, so nDCG collapses to the discount
    itself: 1 / log2(rank + 1).
    """
    rel = {"a": 3}
    at1 = ndcg_at_k(["a", "x", "y"], rel, k=3)
    at2 = ndcg_at_k(["x", "a", "y"], rel, k=3)
    at3 = ndcg_at_k(["x", "y", "a"], rel, k=3)

    assert at1 == pytest.approx(1.0)
    assert at2 == pytest.approx(0.6309297535714574, abs=1e-12)  # 1 / log2(3)
    assert at3 == pytest.approx(0.5, abs=1e-12)  # 1 / log2(4)
    assert at1 > at2 > at3


def test_ndcg_respects_the_cutoff() -> None:
    """A relevant chunk past ``k`` contributes nothing; nDCG@k only sees the top k."""
    assert ndcg_at_k(["x", "y", "z", "a"], {"a": 3}, k=3) == pytest.approx(0.0)
    assert ndcg_at_k(["x", "y", "z", "a"], {"a": 3}, k=4) == pytest.approx(
        0.4306765580733931, abs=1e-12
    )  # 1 / log2(5)


def test_ndcg_graded_relevance_beats_binary_treatment() -> None:
    """The whole reason stage 05 labels 0-3 instead of relevant/not.

    Two chunks: ``a`` answers the question (grade 3), ``c`` is the right paper and wrong
    section (grade 1). Under a binary scheme both are simply "relevant" and the two
    orderings ``[a, c]`` and ``[c, a]`` are indistinguishable — both would score 1.0.

    Under graded gain:
        ideal      = 7/1 + 1/log2(3) = 7.630929753571457
        DCG([c,a]) = 1/1 + 7/log2(3) = 5.416508275000202
        nDCG       = 0.7098097413968655
    """
    rel = {"a": 3, "c": 1}
    best_first = ndcg_at_k(["a", "c"], rel, k=2)
    worst_first = ndcg_at_k(["c", "a"], rel, k=2)

    assert best_first == pytest.approx(1.0)
    assert worst_first == pytest.approx(0.7098097413968655, abs=1e-12)
    assert worst_first < best_first

    # And the binary control: had both been graded 1, the two orders would tie at 1.0.
    binary = {"a": 1, "c": 1}
    assert ndcg_at_k(["a", "c"], binary, k=2) == pytest.approx(
        ndcg_at_k(["c", "a"], binary, k=2)
    )


def test_ndcg_missing_relevant_chunk_costs_exactly_its_gain() -> None:
    """Dropping the grade-2 chunk from rank 2 and keeping the rest.

    DCG([3, 0, 2]) = 7/1 + 0 + 3/log2(4) = 8.5
    nDCG = 8.5 / 9.392789260714372 = 0.9049495058460971
    """
    assert ndcg_at_k(["a", "zzz", "b"], REL, k=3) == pytest.approx(
        0.9049495058460971, abs=1e-12
    )


def test_ndcg_empty_result_list_is_zero_not_nan() -> None:
    """Returning nothing for an answerable query is a failure, and must be scored as one."""
    assert ndcg_at_k([], REL, k=5) == 0.0


def test_ndcg_with_no_labels_is_nan() -> None:
    """A query with no judgments cannot be scored.

    The notebook prototype returned 0.0 here, which is the dangerous answer: it is
    indistinguishable from a total retrieval failure and it drags the mean down for a
    reason that has nothing to do with the retriever.
    """
    assert math.isnan(ndcg_at_k(["a", "b"], {}, k=5))


def test_ndcg_with_only_zero_grade_labels_is_nan() -> None:
    """All labels graded 0 means nothing was findable; there is no ideal ranking."""
    assert math.isnan(ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, k=5))


# ======================================================================================
# MRR
# ======================================================================================


@pytest.mark.parametrize(
    ("ranked", "expected"),
    [
        (["a", "b", "c"], 1.0),  # hit at rank 1
        (["x", "a", "c"], 0.5),  # hit at rank 2
        (["c", "x", "b"], 1.0 / 3.0),  # grade-1 'c' is NOT a hit; 'b' at rank 3 is
        (["x", "y", "z", "a"], 0.25),  # hit at rank 4
    ],
)
def test_mrr_finds_the_first_hit_at_the_right_reciprocal_rank(
    ranked: list[str], expected: float
) -> None:
    assert mrr(ranked, REL) == pytest.approx(expected)


def test_mrr_ignores_grade_one_by_default() -> None:
    """Grade 1 is "right paper, does not answer the question". That is not a hit."""
    assert mrr(["c"], REL) == 0.0  # findable hits exist ('a', 'b') but none returned
    assert mrr(["c"], REL, min_grade=1) == pytest.approx(1.0)


def test_mrr_looks_past_the_top_k() -> None:
    """MRR is computed over the whole list, so "buried at 12" is distinguishable from
    "never found"."""
    ranked = [f"x{i}" for i in range(11)] + ["a"]
    assert mrr(ranked, REL) == pytest.approx(1.0 / 12.0)


def test_mrr_no_hit_returned_is_zero() -> None:
    assert mrr(["x", "y", "z"], REL) == 0.0


def test_mrr_empty_result_list_is_zero() -> None:
    assert mrr([], REL) == 0.0


def test_mrr_with_no_findable_hit_is_nan() -> None:
    """No label at or above min_grade: there was nothing to find, so 0.0 would be a lie."""
    assert math.isnan(mrr(["a", "b"], {"c": 1, "d": 0}))
    assert math.isnan(mrr(["a", "b"], {}))


# ======================================================================================
# recall@k  — the ceiling
# ======================================================================================


def test_recall_counts_only_grade_two_and_above() -> None:
    """REL has two findable chunks ('a' grade 3, 'b' grade 2). 'c' grade 1 does not count."""
    assert recall_at_k(["a", "b"], REL, k=20) == pytest.approx(1.0)
    assert recall_at_k(["a", "c", "d"], REL, k=20) == pytest.approx(0.5)
    assert recall_at_k(["c", "d"], REL, k=20) == 0.0


def test_recall_respects_the_cutoff() -> None:
    """The point of recall@20 is the candidate set, so what falls past k is lost."""
    ranked = ["x", "y", "a", "b"]
    assert recall_at_k(ranked, REL, k=2) == 0.0
    assert recall_at_k(ranked, REL, k=3) == pytest.approx(0.5)
    assert recall_at_k(ranked, REL, k=4) == pytest.approx(1.0)


def test_recall_ignores_duplicate_results() -> None:
    """A retriever that returns the same chunk twice has not recalled twice as much."""
    assert recall_at_k(["a", "a", "a"], REL, k=20) == pytest.approx(0.5)


def test_recall_empty_result_list_is_zero() -> None:
    assert recall_at_k([], REL, k=20) == 0.0


def test_recall_with_no_relevant_chunks_is_nan_and_skipped_not_zero() -> None:
    """Documented choice: NaN, and the query drops out of the mean.

    Recall over an empty target set is undefined. 0.0 would punish the retriever for a
    gap in the *labels*; 1.0 would flatter it; either way the mean silently moves for a
    reason unrelated to retrieval quality. NaN plus a per-metric ``n`` in the summary
    keeps the skip visible.
    """
    assert math.isnan(recall_at_k(["a", "b"], {"c": 1, "d": 0}, k=20))
    assert math.isnan(recall_at_k(["a", "b"], {}, k=20))


# ======================================================================================
# citation precision
# ======================================================================================


def test_citation_precision_over_a_full_top_k() -> None:
    """3 of 5 cited chunks are supported (grade >= 2)."""
    rel = {"a": 3, "b": 2, "e": 2, "c": 1}
    assert citation_precision(["a", "b", "e", "c", "z"], rel, k=5) == pytest.approx(0.6)


def test_citation_precision_when_fewer_than_k_results_are_returned() -> None:
    """Denominator is what was actually cited, not the nominal k.

    Three results, two supported -> 2/3 = 0.6666..., NOT 2/5 = 0.4. Citing two good
    chunks out of a three-item list is not 40% correct citation behaviour.
    """
    assert citation_precision(["a", "b", "zzz"], REL, k=5) == pytest.approx(2.0 / 3.0)


def test_citation_precision_single_lucky_hit_scores_one() -> None:
    """The documented cost of the denominator choice, pinned so nobody is surprised.

    A retriever that returns one correct chunk and nothing else scores 1.0 here. This is
    exactly why the gate never reads citation precision alone: recall@20 on the same run
    is 0.5, and the gate requires both.
    """
    assert citation_precision(["a"], REL, k=5) == pytest.approx(1.0)
    assert recall_at_k(["a"], REL, k=20) == pytest.approx(0.5)


def test_citation_precision_empty_result_list_is_zero() -> None:
    """Returning nothing must not be an exemption from the metric."""
    assert citation_precision([], REL, k=5) == 0.0


def test_citation_precision_with_no_relevant_labels_is_nan() -> None:
    """Nothing returned could ever have been scored as supported, so do not score it."""
    assert math.isnan(citation_precision(["a", "b"], {"c": 1}, k=5))
    assert math.isnan(citation_precision(["a", "b"], {}, k=5))


def test_citation_precision_ignores_results_past_k() -> None:
    assert citation_precision(["a", "z", "z2", "z3", "z4", "b"], REL, k=5) == pytest.approx(0.2)


# ======================================================================================
# the harness
# ======================================================================================


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def tiny_set(tmp_path: Path) -> EvalSet:
    """Two queries, hand-labelled, written to disk so loading is exercised too."""
    queries = _write_jsonl(
        tmp_path / "q.jsonl",
        [
            {"query_id": "t1", "query": "question one"},
            {"query_id": "t2", "query": "question two"},
        ],
    )
    qrels = _write_jsonl(
        tmp_path / "r.jsonl",
        [
            {"query_id": "t1", "chunk_id": "a", "grade": 3, "labeler": "test"},
            {"query_id": "t1", "chunk_id": "b", "grade": 2, "labeler": "test"},
            {"query_id": "t2", "chunk_id": "m", "grade": 3, "labeler": "test"},
        ],
    )
    return load_eval_set(queries, qrels, min_queries=0)


def test_load_eval_set_validates_and_indexes(tiny_set: EvalSet) -> None:
    assert len(tiny_set) == 2
    assert tiny_set.rel_by_query == {"t1": {"a": 3, "b": 2}, "t2": {"m": 3}}


def test_load_eval_set_rel_by_query_is_not_a_defaultdict(tiny_set: EvalSet) -> None:
    """A defaultdict (the notebook's shape) turns a query_id typo into a silent zero."""
    with pytest.raises(KeyError):
        tiny_set.rel_by_query["typo"]


def test_load_eval_set_fails_loudly_on_a_bad_grade(tmp_path: Path) -> None:
    """Grade 7 is off the 0-3 scale. This must raise, not skip the row."""
    q = _write_jsonl(tmp_path / "q.jsonl", [{"query_id": "t1", "query": "x"}])
    r = _write_jsonl(
        tmp_path / "r.jsonl",
        [
            {"query_id": "t1", "chunk_id": "a", "grade": 3, "labeler": "test"},
            {"query_id": "t1", "chunk_id": "b", "grade": 7, "labeler": "test"},
        ],
    )
    with pytest.raises(EvalSetError, match=r"r\.jsonl:2"):
        load_eval_set(q, r, min_queries=0)


def test_load_eval_set_fails_loudly_on_malformed_json(tmp_path: Path) -> None:
    q = tmp_path / "q.jsonl"
    q.write_text('{"query_id": "t1", "query": "x"}\n{not json\n', encoding="utf-8")
    r = _write_jsonl(tmp_path / "r.jsonl", [])
    with pytest.raises(EvalSetError, match="not valid JSON"):
        load_eval_set(q, r, min_queries=0, require_positive_label=False)


def test_load_eval_set_rejects_an_orphan_qrel(tmp_path: Path) -> None:
    """A judgment for a query that does not exist means the two files drifted apart."""
    q = _write_jsonl(tmp_path / "q.jsonl", [{"query_id": "t1", "query": "x"}])
    r = _write_jsonl(
        tmp_path / "r.jsonl",
        [
            {"query_id": "t1", "chunk_id": "a", "grade": 3, "labeler": "test"},
            {"query_id": "GHOST", "chunk_id": "a", "grade": 3, "labeler": "test"},
        ],
    )
    with pytest.raises(EvalSetError, match="unknown query_id"):
        load_eval_set(q, r, min_queries=0)


def test_load_eval_set_rejects_a_query_with_no_positive_label(tmp_path: Path) -> None:
    """Notebook 05 exit check: such a query measures nothing and lowers every mean."""
    q = _write_jsonl(tmp_path / "q.jsonl", [{"query_id": "t1", "query": "x"}])
    r = _write_jsonl(
        tmp_path / "r.jsonl",
        [{"query_id": "t1", "chunk_id": "a", "grade": 0, "labeler": "test"}],
    )
    with pytest.raises(EvalSetError, match="no chunk graded above 0"):
        load_eval_set(q, r, min_queries=0)


def test_load_eval_set_enforces_the_minimum_size(tmp_path: Path) -> None:
    """The gate refuses to run on a set too small for a stable nDCG."""
    q = _write_jsonl(tmp_path / "q.jsonl", [{"query_id": "t1", "query": "x"}])
    r = _write_jsonl(
        tmp_path / "r.jsonl",
        [{"query_id": "t1", "chunk_id": "a", "grade": 3, "labeler": "test"}],
    )
    with pytest.raises(EvalSetError, match="below the configured minimum"):
        load_eval_set(q, r, min_queries=40)


def test_evaluate_scores_a_perfect_retriever(tiny_set: EvalSet) -> None:
    def perfect(query: str, candidate_k: int = 20) -> list[dict[str, str]]:
        return [{"chunk_id": c} for c in ({"question one": ["a", "b"], "question two": ["m"]}[query])]

    per_query, summary = evaluate(perfect, k=5, candidate_k=20, eval_set=tiny_set)
    names = metric_names(5, 20)

    assert list(per_query["query_id"]) == ["t1", "t2"]
    assert summary[names["ndcg"]] == pytest.approx(1.0)
    assert summary[names["mrr"]] == pytest.approx(1.0)
    assert summary[names["recall"]] == pytest.approx(1.0)
    assert summary[names["citation"]] == pytest.approx(1.0)
    assert summary["n_queries"] == 2


def test_evaluate_means_are_hand_computable(tiny_set: EvalSet) -> None:
    """t1 returns [b, a] (swapped), t2 returns nothing.

    t1: nDCG@5 = DCG([2,3]) / DCG([3,2])
               = (3 + 7/log2(3)) / (7 + 3/log2(3))
               = 7.416508275000202 / 8.892789260714372
               = 0.8339912...
        MRR = 1.0 (grade-2 'b' at rank 1 is a hit)
        recall@20 = 1.0 ; citation precision = 2/2 = 1.0
    t2: empty result list -> 0.0 on all four.

    Means: nDCG 0.41700..., MRR 0.5, recall 0.5, citation 0.5.
    """
    def swapped(query: str, candidate_k: int = 20) -> list[str]:
        return {"question one": ["b", "a"], "question two": []}[query]

    _, summary = evaluate(swapped, k=5, candidate_k=20, eval_set=tiny_set)
    names = metric_names(5, 20)

    t1_ndcg = (3 + 7 / math.log2(3)) / (7 + 3 / math.log2(3))
    assert t1_ndcg == pytest.approx(0.8339912, abs=1e-6)
    assert summary[names["ndcg"]] == pytest.approx(t1_ndcg / 2, abs=1e-9)
    assert summary[names["mrr"]] == pytest.approx(0.5)
    assert summary[names["recall"]] == pytest.approx(0.5)
    assert summary[names["citation"]] == pytest.approx(0.5)


def test_evaluate_skips_unscoreable_queries_rather_than_scoring_them_zero(
    tmp_path: Path,
) -> None:
    """One scoreable query, one whose only label is grade 1.

    Recall is NaN for the second, so the mean is taken over one query and stays 1.0.
    Had the unscoreable query been counted as 0.0 (the notebook's behaviour) the mean
    would read 0.5 and the baseline would look half as good as it is.
    """
    q = _write_jsonl(
        tmp_path / "q.jsonl",
        [
            {"query_id": "t1", "query": "one"},
            {"query_id": "t2", "query": "two"},
        ],
    )
    r = _write_jsonl(
        tmp_path / "r.jsonl",
        [
            {"query_id": "t1", "chunk_id": "a", "grade": 3, "labeler": "test"},
            {"query_id": "t2", "chunk_id": "z", "grade": 1, "labeler": "test"},
        ],
    )
    eval_set = load_eval_set(q, r, min_queries=0)
    names = metric_names(5, 20)

    _, summary = evaluate(
        lambda query, candidate_k=20: (["a"] if query == "one" else ["z"]),
        eval_set=eval_set,
    )
    assert summary[names["recall"]] == pytest.approx(1.0)
    assert summary[f"n_{names['recall']}"] == 1.0
    assert summary["n_queries"] == 2


def test_evaluate_accepts_several_result_shapes(tiny_set: EvalSet) -> None:
    """Bare ids, the notebook's chunk dicts, and a nested {"chunk": {...}} all work."""
    names = metric_names(5, 20)
    shapes = [
        lambda q, candidate_k=20: ["a"] if q == "question one" else ["m"],
        lambda q, candidate_k=20: [{"chunk_id": "a" if q == "question one" else "m"}],
        lambda q, candidate_k=20: [{"chunk": {"chunk_id": "a" if q == "question one" else "m"}}],
    ]
    for fn in shapes:
        _, summary = evaluate(fn, eval_set=tiny_set)
        assert summary[names["mrr"]] == pytest.approx(1.0)


def test_evaluate_tolerates_a_search_fn_without_candidate_k(tiny_set: EvalSet) -> None:
    def simple(query: str) -> list[str]:
        return ["a"] if query == "question one" else ["m"]

    _, summary = evaluate(simple, eval_set=tiny_set)
    assert summary[metric_names(5, 20)["mrr"]] == pytest.approx(1.0)


def test_failure_breakdown_separates_recall_from_ranking_failures(tmp_path: Path) -> None:
    """The split notebook 06 insists on, because the two have opposite fixes."""
    q = _write_jsonl(
        tmp_path / "q.jsonl",
        [
            {"query_id": "recall_fail", "query": "one"},
            {"query_id": "rank_fail", "query": "two"},
        ],
    )
    r = _write_jsonl(
        tmp_path / "r.jsonl",
        [
            {"query_id": "recall_fail", "chunk_id": "a", "grade": 3, "labeler": "test"},
            {"query_id": "rank_fail", "chunk_id": "m", "grade": 3, "labeler": "test"},
        ],
    )
    eval_set = load_eval_set(q, r, min_queries=0)

    # 'one' never retrieves 'a' at all; 'two' retrieves 'm' but only at rank 10.
    def search(query: str, candidate_k: int = 20) -> list[str]:
        if query == "one":
            return [f"x{i}" for i in range(20)]
        return [f"y{i}" for i in range(9)] + ["m"]

    per_query, _ = evaluate(search, k=5, candidate_k=20, eval_set=eval_set)
    breakdown = failure_breakdown(per_query, k=5, candidate_k=20)

    assert breakdown["recall_failures"] == ["recall_fail"]
    assert breakdown["ranking_failures"] == ["rank_fail"]


# ======================================================================================
# compare() — the gate
# ======================================================================================

THRESHOLDS = {
    "retrieval": {
        "ndcg_at_5": {"min": 0.55, "regression_tolerance": 0.02},
        "mrr": {"min": 0.60, "regression_tolerance": 0.02},
        "recall_at_20": {"min": 0.85},
        "citation_precision": {"min": 0.80},
    },
    "policy": {"fail_on_missing_metric": True, "compare_against": "champion"},
}

CHAMPION = {
    "ndcg_at_5": 0.61,
    "mrr": 0.66,
    "recall_at_20": 0.88,
    "citation_precision": 0.84,
}


def test_compare_accepts_a_passing_candidate() -> None:
    """Above every floor and an improvement on the champion."""
    candidate = {
        "ndcg_at_5": 0.64,
        "mrr": 0.67,
        "recall_at_20": 0.89,
        "citation_precision": 0.85,
    }
    verdict = compare(CHAMPION, candidate, THRESHOLDS)

    assert isinstance(verdict, CompareVerdict)
    assert verdict.passed
    assert bool(verdict) is True
    assert verdict.failures == ()
    assert verdict.compare_against == "champion"


def test_compare_accepts_a_tiny_regression_inside_tolerance() -> None:
    """0.61 -> 0.60 is a drop of 0.01, inside the 0.02 run-to-run noise tolerance."""
    candidate = dict(CHAMPION, ndcg_at_5=0.60)
    verdict = compare(CHAMPION, candidate, THRESHOLDS)
    assert verdict.passed


def test_compare_rejects_a_candidate_below_the_floor() -> None:
    """nDCG@5 of 0.50 against a floor of 0.55. Fails on the floor rule alone."""
    candidate = dict(CHAMPION, ndcg_at_5=0.50)
    verdict = compare(CHAMPION, candidate, THRESHOLDS)

    assert not verdict.passed
    assert [c.metric for c in verdict.failures] == ["ndcg_at_5"]
    failure = verdict.failures[0]
    assert failure.value == pytest.approx(0.50)
    assert failure.floor == pytest.approx(0.55)
    assert failure.floor_shortfall == pytest.approx(0.05, abs=1e-9)
    assert "below floor" in failure.reasons[0]


def test_compare_rejects_a_small_regression_that_still_clears_the_floor() -> None:
    """The case fixed thresholds alone cannot catch.

    Champion nDCG@5 0.61 -> candidate 0.58. That is comfortably above the 0.55 floor,
    so a floor-only gate would wave it through; it is a drop of 0.03 against a 0.02
    tolerance, so the regression rule catches it. Without this rule a sequence of
    "still above the floor" regressions walks the system down to the floor and parks
    there.
    """
    candidate = dict(CHAMPION, ndcg_at_5=0.58)
    verdict = compare(CHAMPION, candidate, THRESHOLDS)

    assert not verdict.passed
    failure = verdict.failures[0]
    assert failure.metric == "ndcg_at_5"
    assert failure.floor_shortfall == pytest.approx(0.0)  # the floor was cleared
    assert failure.delta == pytest.approx(-0.03, abs=1e-9)
    assert failure.regression_excess == pytest.approx(0.01, abs=1e-9)
    assert "regressed" in failure.reasons[0]


def test_compare_reports_every_failing_metric_not_just_the_first() -> None:
    candidate = {
        "ndcg_at_5": 0.40,
        "mrr": 0.30,
        "recall_at_20": 0.50,
        "citation_precision": 0.85,
    }
    verdict = compare(CHAMPION, candidate, THRESHOLDS)
    assert sorted(c.metric for c in verdict.failures) == ["mrr", "ndcg_at_5", "recall_at_20"]


def test_compare_treats_recall_as_floor_only() -> None:
    """recall@20 has no regression_tolerance: it is the ceiling, and a ranking change
    should not move it. A drop that stays above the floor passes, and is noted."""
    candidate = dict(CHAMPION, recall_at_20=0.86)
    verdict = compare(CHAMPION, candidate, THRESHOLDS)
    assert verdict.passed
    assert any("recall_at_20" in n for n in verdict.notes)


def test_compare_without_a_champion_degrades_to_fixed_thresholds() -> None:
    """The very first run has no champion. Floors still apply; tolerances cannot."""
    verdict = compare(None, dict(CHAMPION, ndcg_at_5=0.56), THRESHOLDS)
    assert verdict.passed
    assert verdict.compare_against == "fixed_thresholds_only"
    assert verdict.notes


def test_compare_honours_fixed_thresholds_only_policy() -> None:
    """With the policy set, a regression past tolerance is deliberately not enforced."""
    cfg = {
        "retrieval": THRESHOLDS["retrieval"],
        "policy": {"fail_on_missing_metric": True, "compare_against": "fixed_thresholds_only"},
    }
    verdict = compare(CHAMPION, dict(CHAMPION, ndcg_at_5=0.56), cfg)
    assert verdict.passed
    assert verdict.compare_against == "fixed_thresholds_only"


def test_compare_fails_on_a_missing_metric() -> None:
    """The quiet killer: a renamed key makes a gate stop checking without saying so."""
    candidate = {k: v for k, v in CHAMPION.items() if k != "citation_precision"}
    verdict = compare(CHAMPION, candidate, THRESHOLDS)

    assert not verdict.passed
    assert verdict.missing_metrics == ("citation_precision",)
    assert verdict.failures[0].value is None


def test_compare_treats_nan_as_missing() -> None:
    candidate = dict(CHAMPION, mrr=float("nan"))
    verdict = compare(CHAMPION, candidate, THRESHOLDS)
    assert not verdict.passed
    assert verdict.missing_metrics == ("mrr",)


def test_compare_can_be_told_not_to_fail_on_a_missing_metric() -> None:
    cfg = {
        "retrieval": THRESHOLDS["retrieval"],
        "policy": {"fail_on_missing_metric": False, "compare_against": "champion"},
    }
    candidate = {k: v for k, v in CHAMPION.items() if k != "citation_precision"}
    verdict = compare(CHAMPION, candidate, cfg)
    assert verdict.passed
    assert verdict.missing_metrics == ("citation_precision",)


def test_compare_verdict_is_serialisable_and_reportable() -> None:
    """Sude's runner needs to put this in a CI annotation, so both forms must work."""
    verdict = compare(CHAMPION, dict(CHAMPION, ndcg_at_5=0.10), THRESHOLDS)
    payload = verdict.as_dict()
    assert payload["passed"] is False
    assert payload["failures"] == ["ndcg_at_5"]
    assert json.dumps(payload)  # must not raise
    assert "FAIL" in verdict.report()


def test_compare_defaults_to_the_real_thresholds_file() -> None:
    """`eval/thresholds_B.yaml` must stay loadable and keep the four retrieval metrics."""
    verdict = compare(None, CHAMPION)
    assert {c.metric for c in verdict.checks} == {
        "ndcg_at_5",
        "mrr",
        "recall_at_20",
        "citation_precision",
    }
    assert verdict.passed


# ======================================================================================
# the shipped example dataset
# ======================================================================================


def test_example_dataset_loads_and_runs_end_to_end() -> None:
    """The worked example must stay runnable; it is the only thing that exercises the
    harness end to end until the real corpus and held-out set exist."""
    eval_set = load_eval_set(
        "eval/datasets/queries_example_B.jsonl",
        "eval/datasets/qrels_example_B.jsonl",
        min_queries=0,
    )
    assert len(eval_set) == 6

    rel = eval_set.rel_by_query

    def oracle(query: str, candidate_k: int = 20) -> list[str]:
        qid = next(q.query_id for q in eval_set.queries if q.query == query)
        return [c for c, _ in sorted(rel[qid].items(), key=lambda kv: -kv[1])][:candidate_k]

    per_query, summary = evaluate(oracle, k=5, candidate_k=20, eval_set=eval_set)
    names = metric_names(5, 20)

    assert len(per_query) == 6
    # An oracle that returns the labels in grade order is the ideal ranking by
    # construction, so every metric pins at 1.0. This is the harness's self-check.
    assert summary[names["ndcg"]] == pytest.approx(1.0)
    assert summary[names["mrr"]] == pytest.approx(1.0)
    assert summary[names["recall"]] == pytest.approx(1.0)
    # The oracle returns every labelled chunk, and only grade >= 2 counts as a supported
    # citation. Per query, supported/returned: ex001 2/4, ex002 2/3, ex003 2/3,
    # ex004 1/3, ex005 2/4, ex006 1/2.
    assert summary[names["citation"]] == pytest.approx((2/4 + 2/3 + 2/3 + 1/3 + 2/4 + 1/2) / 6)


def test_real_eval_files_are_still_empty_and_must_not_be_written_by_this_harness() -> None:
    """Guard rail. `queries_B.jsonl`/`qrels_B.jsonl` are Buse's hand-labelled files and
    stay empty until she labels them; nothing here may populate them."""
    for name in ("queries_B.jsonl", "qrels_B.jsonl"):
        path = Path("eval/datasets") / name
        if path.exists() and path.stat().st_size > 0:
            pytest.skip(f"{name} has been hand-labelled; this guard no longer applies")
