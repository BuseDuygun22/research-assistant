"""Judge calibration against human labels (Sude).

Two separate jobs that are easy to conflate:

**Agreement** — does the judge assign the same labels a human would? Answered by
Cohen's kappa against Buse's qrels, read alongside the confusion matrix. This is
the guardrail on the guardrail: a judge can be highly self-consistent and
consistently wrong, and only comparison against humans catches that. Kappa alone
does not catch it either — on a 0-3 scale where grade 0 dominates, prevalence
distorts kappa in both directions, which is why `agreement_report` returns the
matrix and the disagreements, not just the number.

**Calibration** — when the judge says 0.8, is it right 80% of the time? Answered
by temperature scaling on held-out confidences. An uncalibrated confidence is not
a threshold anyone can defend, which is why `escalation_confidence` ships at 0.0
and should stay there until this module has been run against real labels.

Stdlib only. The whole point is that this can run in the gate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import exp, log

# --- agreement ----------------------------------------------------------------


@dataclass(frozen=True)
class AgreementReport:
    """Kappa plus everything needed to know whether to believe it."""

    kappa: float
    observed_agreement: float
    expected_agreement: float
    n: int
    matrix: dict[tuple[int, int], int]
    """(human, judge) -> count. The diagnostic kappa compresses away."""
    human_prevalence: dict[int, float]
    judge_prevalence: dict[int, float]

    @property
    def interpretation(self) -> str:
        """Landis-Koch bands, with the caveat attached.

        The bands are conventional, not principled, and the architecture doc's
        kappa < 0.5 tripwire sits inside 'moderate'. Read the matrix.
        """
        k = self.kappa
        if k < 0.0:
            return "worse than chance — check for a label-order bug before anything else"
        if k < 0.20:
            return "slight"
        if k < 0.40:
            return "fair"
        if k < 0.60:
            return "moderate"
        if k < 0.80:
            return "substantial"
        return "almost perfect"

    @property
    def is_degenerate(self) -> bool:
        """True when one grade dominates enough that kappa is unstable.

        The case that breaks naive reading: if 95% of qrels are grade 0, a judge
        that always answers 0 scores high observed agreement and a kappa near
        zero, and neither number tells you it has learned nothing. Reported
        explicitly so nobody has to notice it.
        """
        return any(p > 0.8 for p in self.human_prevalence.values())

    def biggest_disagreements(self, limit: int = 5) -> list[tuple[int, int, int]]:
        """(human, judge, count), worst first. Where the rubric is ambiguous."""
        off = [(h, j, n) for (h, j), n in self.matrix.items() if h != j]
        off.sort(key=lambda x: x[2], reverse=True)
        return off[:limit]

    def render(self) -> str:
        lines = [
            f"kappa {self.kappa:.3f} ({self.interpretation})  n={self.n}",
            f"  observed agreement {self.observed_agreement:.3f}, "
            f"expected {self.expected_agreement:.3f}",
        ]
        if self.is_degenerate:
            lines.append(
                "  WARNING: one grade holds >80% of the human labels. Kappa is "
                "unstable here and a constant-output judge can look reasonable."
            )
        lines.append("  worst disagreements (human -> judge):")
        for h, j, n in self.biggest_disagreements():
            lines.append(f"    {h} -> {j}: {n}")
        return "\n".join(lines)


def cohens_kappa(human: Sequence[int], judge: Sequence[int]) -> AgreementReport:
    """Chance-corrected agreement between two raters on the same items.

    Unweighted, per current guidance for nominal labels. On our 0-3 scale that is
    a deliberate conservative choice: unweighted kappa treats a 0-vs-3 error and a
    2-vs-3 error as equally bad, which understates a judge that is merely
    imprecise. If the matrix shows disagreements clustering on adjacent grades,
    quadratic-weighted kappa is the fairer number — but read the matrix first
    rather than reaching for the weighting that flatters.
    """
    if len(human) != len(judge):
        raise ValueError(f"unequal label counts: {len(human)} vs {len(judge)}")
    n = len(human)
    if n == 0:
        raise ValueError("no labels to compare")

    matrix: dict[tuple[int, int], int] = Counter(zip(human, judge, strict=True))
    observed = sum(c for (h, j), c in matrix.items() if h == j) / n

    h_counts = Counter(human)
    j_counts = Counter(judge)
    labels = set(h_counts) | set(j_counts)
    expected = sum((h_counts[x] / n) * (j_counts[x] / n) for x in labels)

    kappa = 1.0 if expected >= 1.0 else (observed - expected) / (1.0 - expected)
    return AgreementReport(
        kappa=kappa,
        observed_agreement=observed,
        expected_agreement=expected,
        n=n,
        matrix=dict(matrix),
        human_prevalence={x: h_counts[x] / n for x in labels},
        judge_prevalence={x: j_counts[x] / n for x in labels},
    )


def krippendorff_alpha_nominal(ratings: Sequence[Sequence[int | None]]) -> float:
    """Agreement across *more than two* annotators, with gaps allowed.

    Kappa handles two raters on complete data. Once Buse double-labels a subset —
    and principle 23 says she must, because you cannot know whether the labels the
    judge is calibrated against are themselves reliable from a single annotator —
    the sensible measure across three or more raters with partial coverage is
    alpha, which tolerates missing cells.

    `ratings` is items x annotators; `None` means that annotator did not label it.
    """
    units = [[r for r in row if r is not None] for row in ratings]
    units = [u for u in units if len(u) >= 2]
    if not units:
        raise ValueError("need at least one item with two or more ratings")

    total_pairable = sum(len(u) for u in units)
    observed = 0.0
    for u in units:
        m = len(u)
        same = sum(c * (c - 1) for c in Counter(u).values())
        observed += same / (m - 1)
    observed /= total_pairable

    all_values = [v for u in units for v in u]
    counts = Counter(all_values)
    n = len(all_values)
    expected = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1)) if n > 1 else 0.0

    return 1.0 if expected >= 1.0 else (observed - expected) / (1.0 - expected)


# --- calibration --------------------------------------------------------------


@dataclass
class TemperatureScaler:
    """Maps raw judge confidences onto probabilities that mean what they say.

    A model asked for a confidence produces a number that correlates with
    correctness and is almost never calibrated — typically overconfident, and
    systematically so. One parameter fixes most of it, which is the reason to use
    temperature scaling rather than anything more elaborate: with the few hundred
    labelled examples we will realistically have, a richer model would fit noise.

    `temperature > 1` means the judge was overconfident and is being damped.
    """

    temperature: float = 1.0
    fitted_on: int = 0

    def apply(self, confidence: float) -> float:
        """Scale one confidence. Identity when unfitted."""
        p = min(max(confidence, 1e-6), 1 - 1e-6)
        logit = log(p / (1 - p)) / self.temperature
        return 1.0 / (1.0 + exp(-logit))

    def fit(
        self, confidences: Sequence[float], correct: Sequence[bool], *, steps: int = 200
    ) -> TemperatureScaler:
        """Fit by minimising negative log-likelihood over a coarse grid.

        A grid rather than a gradient method on purpose: one bounded parameter,
        a few hundred points, and no dependency on an optimiser. It is not the
        fastest way and it is the easiest to verify.
        """
        if len(confidences) != len(correct):
            raise ValueError("confidences and outcomes must be the same length")
        if not confidences:
            raise ValueError("nothing to fit")

        best_t, best_nll = 1.0, float("inf")
        for i in range(1, steps + 1):
            t = i * 10.0 / steps  # search (0, 10]
            nll = 0.0
            for c, ok in zip(confidences, correct, strict=True):
                p = min(max(_scaled(c, t), 1e-9), 1 - 1e-9)
                nll -= log(p) if ok else log(1 - p)
            if nll < best_nll:
                best_t, best_nll = t, nll
        self.temperature = best_t
        self.fitted_on = len(confidences)
        return self


def _scaled(confidence: float, temperature: float) -> float:
    p = min(max(confidence, 1e-6), 1 - 1e-6)
    return 1.0 / (1.0 + exp(-log(p / (1 - p)) / temperature))


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Confidence minus accuracy. Positive is overconfident."""
        return self.mean_confidence - self.accuracy


@dataclass(frozen=True)
class ReliabilityReport:
    bins: list[ReliabilityBin] = field(default_factory=list)
    expected_calibration_error: float = 0.0

    def render(self) -> str:
        lines = [f"expected calibration error {self.expected_calibration_error:.3f}"]
        for b in self.bins:
            if b.count == 0:
                continue
            arrow = "over" if b.gap > 0.05 else ("under" if b.gap < -0.05 else "ok")
            lines.append(
                f"  [{b.lower:.1f},{b.upper:.1f})  n={b.count:<4} "
                f"conf={b.mean_confidence:.2f} acc={b.accuracy:.2f}  {arrow}"
            )
        return "\n".join(lines)


def reliability(
    confidences: Sequence[float], correct: Sequence[bool], *, n_bins: int = 10
) -> ReliabilityReport:
    """Bin confidences and compare each bin's mean confidence to its accuracy.

    This is what makes a threshold defensible. Expected calibration error is one
    number and hides direction; the bins say *where* the judge is wrong, and
    overconfidence in the top bin is the failure that matters — it is exactly the
    range `escalation_confidence` would sit in.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and outcomes must be the same length")
    if not confidences:
        raise ValueError("nothing to report on")

    edges = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    bins: list[ReliabilityBin] = []
    ece = 0.0
    total = len(confidences)

    for lo, hi in edges:
        members = [
            (c, ok)
            for c, ok in zip(confidences, correct, strict=True)
            if (lo <= c < hi) or (hi == 1.0 and c == 1.0)
        ]
        if not members:
            bins.append(ReliabilityBin(lo, hi, 0, 0.0, 0.0))
            continue
        mean_conf = sum(c for c, _ in members) / len(members)
        acc = sum(1 for _, ok in members if ok) / len(members)
        bins.append(ReliabilityBin(lo, hi, len(members), mean_conf, acc))
        ece += (len(members) / total) * abs(mean_conf - acc)

    return ReliabilityReport(bins=bins, expected_calibration_error=ece)


def suggest_escalation_threshold(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    target_precision: float = 0.95,
) -> float | None:
    """Lowest threshold at which accepted verdicts reach `target_precision`.

    The number `escalation_confidence` should be set to, derived rather than
    guessed. Returns `None` when no threshold reaches the target — which is a
    real answer, and means the judge is not yet good enough to gate on at that
    precision. Reporting `1.0` instead would escalate everything and look like a
    setting rather than a failure.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and outcomes must be the same length")
    candidates = sorted({round(c, 2) for c in confidences})
    for threshold in candidates:
        accepted = [ok for c, ok in zip(confidences, correct, strict=True) if c >= threshold]
        if not accepted:
            continue
        if sum(accepted) / len(accepted) >= target_precision:
            return threshold
    return None
