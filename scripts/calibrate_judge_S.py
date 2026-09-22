"""Judge calibration against human labels (Sude): the runnable half of the story.

`judge/calibration_S.py` has the statistics (kappa, reliability, threshold search)
since the design review flagged them as needed. Nothing called them: the doc said
"kappa awaits labels" and stopped there, which meant that once Buse's qrels landed,
turning them into a kappa number was still a script someone had to write from
scratch, under time pressure, after the labels already existed. This is that
script, written before it is needed, so the day it *is* needed is a command, not
a debugging session.

    python scripts/calibrate_judge_S.py --qrels eval/datasets/qrels_B.jsonl \\
        --queries eval/datasets/queries_B.jsonl

What it does: for every (query, chunk, human grade) triple in the qrels, it asks
the *currently configured* judge (`RA_JUDGE_BACKEND`) to grade the same chunk
against the same query, then reports:

  - Cohen's kappa between the judge and the human labels, with the confusion
    matrix and the worst disagreements (`judge/calibration_S.py:cohens_kappa`)
  - a reliability diagram: is the judge's confidence a probability that means
    what it says (`reliability`, expected calibration error)
  - the escalation-confidence threshold that would make `passed`-and-accepted
    verdicts hit 95% precision against the human labels (`suggest_escalation_threshold`)

Exit codes mirror the eval gate's, deliberately, since both answer "is this
number safe to promote on": 0 kappa clears the architecture doc's 0.5 tripwire,
1 it does not (rework the rubric before trusting DPO labels from this judge),
2 refused (not enough labels, or the qrels/queries do not line up) — a refusal is
not a verdict on the judge, only a statement that this run cannot produce one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_assistant.contracts.retrieval_J import Chunk  # noqa: E402
from research_assistant.judge.calibration_S import (  # noqa: E402
    cohens_kappa,
    reliability,
    suggest_escalation_threshold,
)
from research_assistant.judge.relevance_S import grade_chunk  # noqa: E402
from research_assistant.mcp_server.backend_S import get_backend  # noqa: E402

KAPPA_TRIPWIRE = 0.5  # docs/architecture_J.md Stage 5: rework the rubric below this.
MIN_LABELS_DEFAULT = 30  # below this, kappa is noise; refuse rather than report it.


class CalibrationRefusal(RuntimeError):
    """Not enough to compute a defensible number, distinct from a bad number."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise CalibrationRefusal(f"{path} does not exist")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CalibrationRefusal(f"{path}:{i}: malformed JSON ({exc})") from exc
    return rows


def load_labelled_pairs(
    qrels_path: Path, queries_path: Path
) -> list[tuple[str, Chunk, int]]:
    """(query text, chunk, human grade) for every qrel row whose chunk resolves.

    A chunk that does not resolve (stale id, wrong corpus version) is dropped, not
    silently counted as agreement or disagreement - it is neither, and folding it
    into the kappa denominator would understate how much real evidence there is.
    """
    qrels = _read_jsonl(qrels_path)
    query_rows = _read_jsonl(queries_path)
    query_text = {r["query_id"]: r["query"] for r in query_rows}

    backend = get_backend()
    pairs: list[tuple[str, Chunk, int]] = []
    missing_query = 0
    missing_chunk = 0
    for row in qrels:
        q = query_text.get(row["query_id"])
        if q is None:
            missing_query += 1
            continue
        chunk = backend.get_chunk(row["chunk_id"])
        if chunk is None:
            missing_chunk += 1
            continue
        pairs.append((q, chunk, int(row["grade"])))

    if missing_query or missing_chunk:
        print(
            f"note: dropped {missing_query} qrel row(s) with no matching query_id "
            f"and {missing_chunk} with a chunk_id not in this corpus_version",
            file=sys.stderr,
        )
    return pairs


def run_judge(pairs: Sequence[tuple[str, Chunk, int]]) -> tuple[list[int], list[int], list[float]]:
    """Grade every pair with the configured judge. Returns (human, judge, confidence)."""
    human: list[int] = []
    judged: list[int] = []
    confidence: list[float] = []
    for query, chunk, human_grade in pairs:
        verdict = grade_chunk(query, chunk)
        human.append(human_grade)
        judged.append(int(verdict.grade))
        confidence.append(verdict.confidence)
    return human, judged, confidence


def render_report(
    human: Sequence[int], judged: Sequence[int], confidence: Sequence[float]
) -> tuple[str, bool]:
    agreement = cohens_kappa(human, judged)
    correct = [h == j for h, j in zip(human, judged, strict=True)]
    rel = reliability(confidence, correct)
    threshold = suggest_escalation_threshold(confidence, correct)

    lines = [
        f"judge calibration — n={len(human)} labelled (query, chunk) pairs",
        "",
        "agreement with human labels (docs/architecture_J.md Stage 5: rework below kappa 0.5)",
        agreement.render(),
        "",
        "confidence reliability (is a judge confidence of 0.8 right 80% of the time?)",
        rel.render(),
        "",
    ]
    if threshold is None:
        lines.append(
            "suggested RA_ESCALATION_CONFIDENCE: none reaches 95% precision — this "
            "judge is not yet good enough to gate escalation on its own confidence"
        )
    else:
        lines.append(f"suggested RA_ESCALATION_CONFIDENCE: {threshold:.2f}")

    clears_tripwire = agreement.kappa >= KAPPA_TRIPWIRE
    lines.append("")
    lines.append(
        f"VERDICT: kappa {agreement.kappa:.3f} "
        + ("clears" if clears_tripwire else "is BELOW")
        + f" the {KAPPA_TRIPWIRE} tripwire — "
        + (
            "safe to use this judge's grades as DPO training signal."
            if clears_tripwire
            else "do NOT train on this judge's grades yet; rework the rubric first."
        )
    )
    return "\n".join(lines), clears_tripwire


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qrels", default="eval/datasets/qrels_B.jsonl")
    ap.add_argument("--queries", default="eval/datasets/queries_B.jsonl")
    ap.add_argument(
        "--min-labels",
        type=int,
        default=MIN_LABELS_DEFAULT,
        help="refuse below this many labelled pairs; kappa is noise on a handful of rows",
    )
    ap.add_argument("--out", default=None, help="also write the raw grades as JSON here")
    args = ap.parse_args(argv)

    try:
        pairs = load_labelled_pairs(Path(args.qrels), Path(args.queries))
        if len(pairs) < args.min_labels:
            raise CalibrationRefusal(
                f"only {len(pairs)} labelled pairs resolved, below --min-labels "
                f"{args.min_labels}; kappa on this few rows would not be defensible"
            )
        human, judged, confidence = run_judge(pairs)
    except CalibrationRefusal as exc:
        print(f"CALIBRATION REFUSED: {exc}")
        return 2

    report, clears_tripwire = render_report(human, judged, confidence)
    print(report)

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {"human": human, "judge": judged, "confidence": confidence}, indent=2
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")

    return 0 if clears_tripwire else 1


if __name__ == "__main__":
    raise SystemExit(main())
