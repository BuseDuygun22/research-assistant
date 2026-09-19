"""Statistical significance for the eval gate (Sude).

The gate's job is to answer one question — did this change make the system better
— and a threshold comparison on two point estimates cannot answer it. Retrieval
metrics vary per query by far more than the effects we are trying to detect, so a
bare `candidate > baseline` test promotes noise roughly half the time it fires on
a no-op change.

Three decisions are encoded here.

**Paired, not unpaired.** [E] Baseline and candidate are scored on the *same*
queries, so the per-query difference cancels query difficulty — the dominant
variance term. This is the single cheapest sensitivity gain available to us: the
same labels from Buse detect a substantially smaller true effect once the
comparison is paired. An unpaired test on the same data throws that away.

**Bootstrap, not a t-test.** [E] nDCG@5 over 5 ranks is bounded, discrete and
sharply non-normal; MRR is worse. The bootstrap makes no distributional
assumption, which matters more here than the efficiency a parametric test would
buy on data that satisfied its assumptions.

**Seeded.** [E] The gate makes a promote/reject decision, and a decision that
cannot be reproduced cannot be appealed. The resampling seed is an explicit
argument, defaulted and logged, never `random.random()`.

Deliberately stdlib-only: no numpy, no scipy. The gate must import and run on a
bare checkout, because a gate with an install step is a gate that breaks on the
one push where it matters.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from statistics import fmean, stdev

# Normal quantiles, hardcoded so this module needs no scipy.
_Z_95 = 1.959964  # two-sided alpha = 0.05
_Z_POWER_80 = 0.841621  # beta = 0.20


class AlignmentError(ValueError):
    """Raised when two runs cannot be paired query-for-query.

    Strict on purpose, exactly as the dataset loader is: silently intersecting
    two runs down to their common queries would shrink the eval set precisely
    when something upstream is broken, which makes the gate easier to pass at the
    moment it should be hardest.
    """


@dataclass(frozen=True)
class PairedResult:
    """The outcome of one paired comparison on one metric."""

    metric: str
    n: int
    baseline_mean: float
    candidate_mean: float
    observed_diff: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    seed: int
    p_value: float = 1.0
    """Bootstrap achieved significance level, two-sided.

    Carried separately from the interval because the two answer different
    questions and only one of them can be corrected for multiple comparisons. The
    interval says how large the effect is; the p-value says how surprising it
    would be under no effect, and Holm needs an orderable quantity. A fixed 95%
    interval has no such quantity — every metric reports the same 0.95 — which is
    why correcting on it silently rejects everything.
    """

    @property
    def significant(self) -> bool:
        """True when the confidence interval excludes zero.

        This is the whole point of the module: a positive `observed_diff` whose
        interval straddles zero is not evidence of improvement, and promoting on
        it is how a gate becomes a rubber stamp.
        """
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def direction(self) -> str:
        if not self.significant:
            return "inconclusive"
        return "improvement" if self.observed_diff > 0 else "regression"

    def summary(self) -> str:
        return (
            f"{self.metric}: {self.baseline_mean:.4f} -> {self.candidate_mean:.4f} "
            f"(diff {self.observed_diff:+.4f}, "
            f"{self.confidence:.0%} CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}], "
            f"n={self.n}, {self.direction})"
        )


def align(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
) -> tuple[list[float], list[float]]:
    """Pair two per-query score maps, or refuse.

    Returns (baseline_scores, candidate_scores) ordered by query id so the
    pairing is stable across runs and the bootstrap seed means the same thing
    twice.
    """
    missing = set(baseline) ^ set(candidate)
    if missing:
        only_base = sorted(set(baseline) - set(candidate))[:5]
        only_cand = sorted(set(candidate) - set(baseline))[:5]
        raise AlignmentError(
            f"runs cover different queries ({len(missing)} unmatched). "
            f"baseline-only: {only_base or '-'}; candidate-only: {only_cand or '-'}. "
            "Both runs must score the identical query set for a paired comparison."
        )
    keys = sorted(baseline)
    return [baseline[k] for k in keys], [candidate[k] for k in keys]


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    metric: str = "metric",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedResult:
    """Percentile bootstrap over the per-query difference.

    Resamples *pairs*, not the two runs independently — resampling the runs
    separately would reintroduce the query-difficulty variance the pairing exists
    to remove.
    """
    if len(baseline) != len(candidate):
        raise AlignmentError(
            f"paired comparison needs equal-length runs, got "
            f"{len(baseline)} and {len(candidate)}"
        )
    n = len(baseline)
    if n < 2:
        raise ValueError(f"need at least 2 paired observations, got {n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")

    diffs = [c - b for b, c in zip(baseline, candidate, strict=True)]
    observed = fmean(diffs)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        # Sample indices, not values, so the pairing survives the resample.
        resample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(fmean(resample))
    means.sort()

    tail = (1.0 - confidence) / 2.0
    lo = means[_percentile_index(n_resamples, tail)]
    hi = means[_percentile_index(n_resamples, 1.0 - tail)]

    # Two-sided achieved significance level: how much of the resampled
    # distribution sits on the far side of zero, doubled. Floored at 1/n_resamples
    # rather than reported as 0 — the bootstrap cannot resolve a p smaller than
    # its own resolution, and printing 0.0 would claim more than it measured.
    below = sum(1 for m in means if m <= 0.0) / n_resamples
    above = sum(1 for m in means if m >= 0.0) / n_resamples
    p_value = min(1.0, max(2.0 * min(below, above), 1.0 / n_resamples))

    return PairedResult(
        metric=metric,
        n=n,
        baseline_mean=fmean(baseline),
        candidate_mean=fmean(candidate),
        observed_diff=observed,
        ci_low=lo,
        ci_high=hi,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
        p_value=p_value,
    )


def _percentile_index(size: int, q: float) -> int:
    """Index into a sorted list of `size` for quantile `q`, clamped in range."""
    idx = int(round(q * (size - 1)))
    return max(0, min(size - 1, idx))


# --- gate decisions ----------------------------------------------------------


@dataclass(frozen=True)
class GateVerdict:
    """A promote/reject decision, with the reason attached.

    `reason` is not decoration. "The gate went red" prompts a re-run; "nDCG@5
    moved +0.006, below the 0.02 we decided matters" prompts a conversation about
    whether the change was worth making.
    """

    promote: bool
    reason: str
    results: tuple[PairedResult, ...]

    def report(self) -> str:
        head = "PROMOTE" if self.promote else "REJECT"
        lines = [f"{head}: {self.reason}"]
        lines.extend("  " + r.summary() for r in self.results)
        return "\n".join(lines)


def practically_significant(result: PairedResult, min_effect: float) -> bool:
    """Statistically significant *and* large enough to care about.

    Two separate questions that a p-value conflates. With enough queries, a
    change of +0.001 nDCG becomes statistically detectable while remaining
    operationally meaningless — and a gate that promotes on it will happily
    accumulate complexity for no benefit. The guard is the *lower* CI bound
    rather than the point estimate: it asserts the effect is *at least*
    `min_effect`, not merely that its centre happens to land above it.
    """
    if min_effect < 0:
        raise ValueError(f"min_effect must be non-negative, got {min_effect}")
    return result.significant and result.ci_low >= min_effect


def holm_adjusted(results: Sequence[PairedResult], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni over several metrics tested at once.

    Testing four metrics at 5% each means roughly a 1-in-5 chance that at least
    one fires on a run where nothing changed. Run on every push, that is a steady
    trickle of false regressions, and a gate that cries wolf is a gate people
    learn to skip — the same failure mode as a gate that costs money.

    Holm rather than plain Bonferroni: uniformly more powerful, no extra
    assumptions, and short enough to read.

    Operates on `p_value`, not on the interval. An earlier version ranked by the
    distance from zero to the nearer CI bound and then compared the *confidence
    level* against the corrected threshold — which rejected every metric whenever
    more than one was tested, because a 95% interval reports 0.95 for all of them
    regardless of the evidence. Correcting a fixed-level interval is not a
    correction; it needs a quantity that varies with the evidence.

    Returns a mask aligned with `results`.
    """
    if not results:
        return []
    m = len(results)
    order = sorted(range(m), key=lambda i: results[i].p_value)
    keep = [False] * m
    for rank, idx in enumerate(order):
        # Holm: the k-th smallest p is tested against alpha / (m - k).
        if results[idx].p_value <= alpha / (m - rank):
            keep[idx] = True
        else:
            break  # Holm stops at the first failure; the rest stay rejected.
    return keep


def decide(
    results: Sequence[PairedResult],
    *,
    min_effect: float = 0.0,
    require_no_regression: bool = True,
) -> GateVerdict:
    """Promote or reject, given per-metric paired comparisons.

    Asymmetric on purpose. A *regression* rejects on statistical significance
    alone, without needing to clear `min_effect`: a change that measurably makes
    something worse should be looked at even when the amount is small. An
    *improvement* must clear both bars. Guarding both directions at the same
    threshold would let a change ship that traded a large regression for a
    marginally larger improvement.
    """
    results = tuple(results)
    if not results:
        return GateVerdict(False, "no metrics compared", results)

    regressions = [r for r in results if r.significant and r.observed_diff < 0]
    if require_no_regression and regressions:
        names = ", ".join(r.metric for r in regressions)
        return GateVerdict(False, f"significant regression in {names}", results)

    improvements = [r for r in results if practically_significant(r, min_effect)]
    if not improvements:
        best = max(results, key=lambda r: r.observed_diff)
        return GateVerdict(
            False,
            f"no metric cleared a {min_effect:+.3f} improvement with the CI excluding "
            f"zero (best: {best.metric} {best.observed_diff:+.4f}, "
            f"CI [{best.ci_low:+.4f}, {best.ci_high:+.4f}])",
            results,
        )
    names = ", ".join(r.metric for r in improvements)
    return GateVerdict(True, f"significant improvement in {names}", results)


# --- sizing the eval set -----------------------------------------------------
#
# These two are the numbers Buse needs *before* she finishes labelling, not
# after. They answer "how many queries buy me the ability to see the effect I
# care about", which is a question about the labelling budget, not about the
# gate.


def required_queries(
    target_effect: float,
    sd_of_differences: float,
    *,
    confidence: float = 0.95,
    power: float = 0.80,
) -> int:
    """Paired query count needed to detect `target_effect` at the given power.

    Normal approximation, which is close enough for a labelling-budget decision
    and transparent enough to argue with. `sd_of_differences` is the standard
    deviation of the per-query *difference*, not of either run's scores — on a
    pilot run of 20-30 queries it is measurable in an afternoon, and guessing it
    is the largest error term in this estimate by far.
    """
    if target_effect <= 0:
        raise ValueError(f"target_effect must be positive, got {target_effect}")
    if sd_of_differences <= 0:
        raise ValueError(f"sd_of_differences must be positive, got {sd_of_differences}")
    z_a = _Z_95 if confidence == 0.95 else _z_two_sided(confidence)
    z_b = _Z_POWER_80 if power == 0.80 else _z_one_sided(power)
    n = ((z_a + z_b) ** 2) * (sd_of_differences**2) / (target_effect**2)
    return max(2, int(n) + (1 if n % 1 else 0))


def minimum_detectable_effect(
    n: int,
    sd_of_differences: float,
    *,
    confidence: float = 0.95,
    power: float = 0.80,
) -> float:
    """The smallest true effect an eval set of `n` queries can reliably see.

    The inverse of `required_queries`, and the more honest one to report: it turns
    "the gate went green" into "the gate can see changes of at least X", which is
    the sentence that belongs in the results section.
    """
    if n < 2:
        raise ValueError(f"need at least 2 paired observations, got {n}")
    if sd_of_differences <= 0:
        raise ValueError(f"sd_of_differences must be positive, got {sd_of_differences}")
    z_a = _Z_95 if confidence == 0.95 else _z_two_sided(confidence)
    z_b = _Z_POWER_80 if power == 0.80 else _z_one_sided(power)
    return (z_a + z_b) * sd_of_differences / sqrt(n)


def observed_sd(baseline: Sequence[float], candidate: Sequence[float]) -> float:
    """Standard deviation of the per-query difference, for feeding the two above."""
    if len(baseline) != len(candidate):
        raise AlignmentError("runs must be equal length")
    if len(baseline) < 2:
        raise ValueError("need at least 2 paired observations")
    return stdev([c - b for b, c in zip(baseline, candidate, strict=True)])


def _z_two_sided(confidence: float) -> float:
    return _inverse_normal_cdf(1.0 - (1.0 - confidence) / 2.0)


def _z_one_sided(power: float) -> float:
    return _inverse_normal_cdf(power)


def _inverse_normal_cdf(p: float) -> float:
    """Acklam's rational approximation; accurate to ~1e-9 over (0, 1).

    Present only so non-default confidence/power values still work without
    dragging scipy into the gate's dependency set.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416)
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = sqrt(-2 * _log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > p_high:
        q = sqrt(-2 * _log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def _log(x: float) -> float:
    from math import log

    return log(x)
