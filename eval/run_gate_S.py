"""The CI eval gate (Sude).

Runs the eval set through the system and decides promote or reject. Five refusals
sit in front of the decision, and each exists because the gate is only worth
having if it cannot be quietly wrong:

1. **A degraded backend refuses to produce a verdict.** `StubRetrieval` reports
   `readiness() == "degraded"`; numbers from it are not publishable, so the gate
   will not promote on them. Overridable with `--allow-degraded` for plumbing
   runs, which is what CI on every push actually does — it checks the pipeline
   executes, not that quality improved.
2. **Stamp mismatch refuses to compare.** Differing `corpus_version`,
   `embedding_model` or `reranker_version` means the two runs are not measuring
   the same thing, and a "+4 nDCG" that was really a corpus change is how a gate
   becomes a rubber stamp.
3. **Too few queries refuses to conclude.** Below the floor the confidence
   interval is wider than any effect worth shipping, so a green gate means
   nothing. The floor is `dataset.min_queries` in Track A's
   `eval/thresholds_B.yaml` (default 50 if that file is absent), so the number
   lives in one place and belongs to whoever owns the eval set.
4. **A malformed dataset row fails the gate.** Skipping bad rows silently shrinks
   the eval set, making the gate easier to pass exactly when the data degrades.
5. **Multiple metrics are Holm-corrected.** Four metrics at 5% each is not a 5%
   false-positive rate; run per-PR that is a steady trickle of false regressions,
   and a gate that cries wolf gets skipped.

Retrieval metrics (nDCG, MRR) come from Buse's `eval/metrics/retrieval_B.py`,
bound lazily. While that file is empty the gate runs on generation metrics alone
and says so, rather than failing to import.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from research_assistant.config_J import EVAL_DIR, get_settings
from research_assistant.contracts.eval_dataset_J import EvalDataset, EvalQuery, load_eval_dataset
from research_assistant.mcp_server.health_S import readiness

from .metrics.generation_S import (
    answer_relevance,
    claim_granularity,
    context_recall,
    score_abstention,
    trajectory_metrics,
)
from .metrics.significance_S import (
    GateVerdict,
    PairedResult,
    align,
    decide,
    holm_adjusted,
    minimum_detectable_effect,
    observed_sd,
    paired_bootstrap,
)

logger = logging.getLogger(__name__)

MIN_QUERIES = 50
"""Default floor, used when the thresholds file does not set one. Below the floor
the interval is wider than any effect worth shipping. A gate that cannot resolve
the difference it is asked about should say so, not go green."""


def query_floor(thresholds_path: Path | None) -> int:
    """The minimum eval-set size, from Track A's thresholds file when it says.

    A missing file, an unreadable one, or a value that is not a positive integer
    all fall back to `MIN_QUERIES` rather than raising: a malformed thresholds file
    must not be able to switch the floor off, and it must not crash the gate.
    """
    if thresholds_path is None or not thresholds_path.exists():
        return MIN_QUERIES
    try:
        raw = yaml.safe_load(thresholds_path.read_text(encoding="utf-8")) or {}
        value = raw.get("dataset", {}).get("min_queries")
    except (OSError, yaml.YAMLError, AttributeError):
        return MIN_QUERIES
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 \
        else MIN_QUERIES


class GateRefusal(RuntimeError):
    """The gate declined to produce a verdict. Distinct from a reject: a reject
    is an answer, a refusal means the question could not be asked."""


@dataclass
class RunResult:
    """One system's scores over the eval set, plus the stamps that make them
    comparable to another run."""

    label: str
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    trajectories: list[list[dict[str, Any]]] = field(default_factory=list)
    abstentions: list[tuple[bool, bool]] = field(default_factory=list)
    # Reported, never gated: a diagnostic is not a quality metric, and adding it
    # to the Holm family would dilute the correction for the metrics that are.
    diagnostics: dict[str, dict[str, float]] = field(default_factory=dict)
    corpus_version: str = ""
    embedding_model: str = ""
    reranker_version: str = ""

    @property
    def stamps(self) -> tuple[str, str, str]:
        return (self.corpus_version, self.embedding_model, self.reranker_version)

    def metric(self, name: str) -> dict[str, float]:
        return {qid: scores[name] for qid, scores in self.per_query.items() if name in scores}

    @property
    def metric_names(self) -> list[str]:
        names: set[str] = set()
        for scores in self.per_query.values():
            names.update(scores)
        return sorted(names)


def _retrieval_metrics() -> Any | None:
    """Bind Buse's retrieval metrics if they exist yet.

    Lazy for the same reason `TrackARetrieval` is: this module must import while
    her file is still empty, so both tracks can run end to end alone.
    """
    try:
        from eval.metrics import retrieval_B  # noqa: PLC0415

        if not hasattr(retrieval_B, "ndcg_at_k"):
            return None
        return retrieval_B
    except ImportError:
        return None


def evaluate(dataset: EvalDataset, split: str, label: str) -> RunResult:
    """Run every query in `split` and collect per-query scores.

    Imports the graph lazily so `--compare` on two stored result files needs
    neither a backend nor a model.
    """
    from research_assistant.agents.graph_S import run  # noqa: PLC0415

    settings = get_settings()
    result = RunResult(
        label=label,
        corpus_version=settings.corpus_version,
        embedding_model=settings.embedding_model,
        reranker_version="none",
    )
    retrieval = _retrieval_metrics()
    if retrieval is None:
        logger.warning(
            "eval/metrics/retrieval_B.py has no ndcg_at_k yet — running on generation "
            "metrics only. Retrieval quality is NOT being gated."
        )

    queries: Sequence[EvalQuery] = dataset.for_split(split)
    for q in queries:
        state, _ = run(q.query, run_id=q.query_id)
        relevant = dataset.relevant(q.query_id)
        retrieved = [r.chunk.chunk_id for r in state.evidence]

        scores: dict[str, float] = {
            "answer_relevance": answer_relevance(
                [state.answer.relevance] if state.answer else [0]
            ),
            "citation_precision": (
                state.faithfulness.citation_precision if state.faithfulness else 0.0
            ),
            "coverage": state.faithfulness.coverage if state.faithfulness else 0.0,
            "context_recall": context_recall(retrieved, list(relevant)),
        }
        if retrieval is not None:
            scores["ndcg@5"] = retrieval.ndcg_at_k(retrieved, relevant, 5)
            scores["mrr"] = retrieval.mrr(retrieved, relevant)

        result.per_query[q.query_id] = scores
        if state.draft is not None:
            prof = claim_granularity(
                state.draft.text, [c.text for c in state.draft.claims]
            )
            result.diagnostics[q.query_id] = {
                "claims_per_sentence": prof.claims_per_sentence,
                "words_per_claim": prof.mean_words_per_claim,
                "cost_spent": state.budget.cost_spent,
            }
        result.trajectories.append(state.trajectory())
        result.abstentions.append(
            (q.intent == "unanswerable", state.outcome == "abstained")
        )
    return result


def compare(
    baseline: RunResult,
    candidate: RunResult,
    *,
    min_effect: float,
    seed: int,
    n_resamples: int,
    allow_small: bool = False,
    min_queries: int = MIN_QUERIES,
) -> GateVerdict:
    """Paired comparison across every shared metric, Holm-corrected."""
    if baseline.stamps != candidate.stamps:
        raise GateRefusal(
            f"stamps differ — baseline {baseline.stamps} vs candidate "
            f"{candidate.stamps}. These runs are not measuring the same thing, so "
            "a difference between them is not attributable to the change under test."
        )

    shared = [m for m in candidate.metric_names if m in baseline.metric_names]
    if not shared:
        raise GateRefusal("no metrics in common between the two runs")

    n = len(candidate.per_query)
    if n < min_queries and not allow_small:
        raise GateRefusal(
            f"{n} queries is below the {min_queries}-query floor. The confidence "
            f"interval would be wider than any effect worth shipping — a green gate "
            f"here would mean nothing. Pass --allow-small to override for a smoke run."
        )

    results: list[PairedResult] = []
    for name in shared:
        b, c = align(baseline.metric(name), candidate.metric(name))
        results.append(
            paired_bootstrap(
                b, c, metric=name, seed=seed, n_resamples=n_resamples
            )
        )

    # Holm across metrics. A result the correction rejects is reported with its
    # interval widened to include zero, so `decide` treats it as inconclusive
    # rather than significant — the correction changes the verdict, not the
    # measurement.
    kept = holm_adjusted(results)
    corrected = [
        r
        if keep
        else PairedResult(
            metric=r.metric,
            n=r.n,
            baseline_mean=r.baseline_mean,
            candidate_mean=r.candidate_mean,
            observed_diff=r.observed_diff,
            ci_low=min(0.0, r.ci_low),
            ci_high=max(0.0, r.ci_high),
            confidence=r.confidence,
            n_resamples=r.n_resamples,
            seed=r.seed,
            p_value=r.p_value,
        )
        for r, keep in zip(results, kept, strict=True)
    ]
    return decide(corrected, min_effect=min_effect)


def sensitivity_report(baseline: RunResult, candidate: RunResult) -> list[str]:
    """What the eval set can and cannot see, stated before any verdict.

    "The gate went green" is not a finding; "the gate can resolve changes of at
    least 0.04" is. Reported every run so the number stays in front of whoever
    reads the result.
    """
    lines = []
    for name in sorted(set(baseline.metric_names) & set(candidate.metric_names)):
        b, c = align(baseline.metric(name), candidate.metric(name))
        if len(b) < 2:
            continue
        sd = observed_sd(b, c)
        if sd <= 0:
            lines.append(f"  {name}: runs identical, sensitivity undefined")
            continue
        mde = minimum_detectable_effect(len(b), sd)
        lines.append(f"  {name}: n={len(b)}, sd={sd:.4f}, can resolve >= {mde:.4f}")
    return lines


def check_backend(allow_degraded: bool) -> None:
    """Refuse to grade a run served by a stub."""
    status = readiness()
    if status.status != "ok" and not allow_degraded:
        raise GateRefusal(
            f"backend is '{status.status}' ({status.detail}). Numbers from this "
            f"backend are not publishable, so the gate will not promote on them. "
            f"Pass --allow-degraded for a plumbing-only run."
        )
    if status.status != "ok":
        logger.warning(
            "running against a DEGRADED backend (%s) — plumbing only, quality "
            "numbers are meaningless",
            status.detail,
        )


def load_default_dataset() -> EvalDataset:
    return load_eval_dataset(
        EVAL_DIR / "datasets" / "queries_B.jsonl",
        EVAL_DIR / "datasets" / "qrels_B.jsonl",
        get_settings().corpus_version,
    )


def _save(result: RunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "label": result.label,
                "per_query": result.per_query,
                "trajectories": result.trajectories,
                "abstentions": result.abstentions,
                "corpus_version": result.corpus_version,
                "embedding_model": result.embedding_model,
                "reranker_version": result.reranker_version,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _load(path: Path) -> RunResult:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RunResult(
        label=raw["label"],
        per_query=raw["per_query"],
        trajectories=raw.get("trajectories", []),
        abstentions=[tuple(x) for x in raw.get("abstentions", [])],
        corpus_version=raw.get("corpus_version", ""),
        embedding_model=raw.get("embedding_model", ""),
        reranker_version=raw.get("reranker_version", ""),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the eval gate.")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--out", type=Path, help="Write this run's scores here.")
    parser.add_argument("--baseline", type=Path, help="Compare against this stored run.")
    parser.add_argument("--min-effect", type=float, default=0.02)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=EVAL_DIR / "thresholds_B.yaml",
        help="Track A thresholds file. Currently supplies dataset.min_queries; "
        "per-metric min_effect and regression tolerances are not yet consumed.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--allow-degraded", action="store_true")
    parser.add_argument("--allow-small", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        check_backend(args.allow_degraded)
        if args.split == "test" and args.baseline is None:
            # Held-out discipline: the test split exists to report a final number,
            # not to pick between options. Using it to choose is how a held-out
            # set stops being held out.
            logger.warning(
                "running on the TEST split — this must not be used to select a "
                "prompt, a reranker, or a threshold"
            )
        dataset = load_default_dataset()
        current = evaluate(dataset, args.split, label="candidate")
    except GateRefusal as exc:
        print(f"GATE REFUSED: {exc}")
        return 2
    except FileNotFoundError as exc:
        print(f"GATE REFUSED: eval dataset missing ({exc})")
        return 2

    abst = score_abstention(current.abstentions)
    traj = trajectory_metrics(current.trajectories)
    print(f"\n=== {current.label} · split={args.split} · n={len(current.per_query)} ===")
    for name in current.metric_names:
        scores = current.metric(name).values()
        print(f"  {name:<20} {sum(scores) / max(1, len(scores)):.4f}")
    print(f"  abstention recall    {abst.abstention_recall:.4f}")
    print(f"  abstention precision {abst.abstention_precision:.4f}")
    for k, v in sorted(traj.items()):
        print(f"  {k:<20} {v:.4f}")

    if args.out:
        _save(current, args.out)
        print(f"\nwrote {args.out}")

    if not args.baseline:
        print("\nNo baseline given — reported scores only, no promote/reject decision.")
        return 0

    try:
        verdict = compare(
            _load(args.baseline),
            current,
            min_effect=args.min_effect,
            seed=args.seed,
            n_resamples=args.n_resamples,
            allow_small=args.allow_small,
            min_queries=query_floor(args.thresholds),
        )
    except GateRefusal as exc:
        print(f"\nGATE REFUSED: {exc}")
        return 2

    print("\nsensitivity:")
    for line in sensitivity_report(_load(args.baseline), current):
        print(line)
    print()
    print(verdict.report())
    return 0 if verdict.promote else 1


if __name__ == "__main__":
    sys.exit(main())
