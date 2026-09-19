"""Stage 07 hyperparameter sweep: how many usable DPO pairs does each
`min_score_gap` actually produce?

This answers exactly one of the two questions `min_score_gap` needs answered
before it can be trusted (the pair-yield/coverage question). The other —
"how reliable is the preference at each gap, compared to a human?" — needs
Sude's real judge to have actually run, and the third — "which gap trains the
best reranker?" — needs a completed DPO run per gap evaluated against the
held-out qrels. Neither exists yet in this repo, so this script does not
pretend to answer them; it prints exactly what it measured and says what is
still missing.

Two input modes:

* `--judge-scores path/to/real_verdicts.jsonl` — once Sude's judge has scored
  the real candidate pools, point this at that file (one `RelevanceVerdict`
  JSON object per line) for the real answer.
* (default) simulated from `eval/datasets/drafts/llm_draft_grades_B.json` and
  `eval/datasets/drafts/pool_B.jsonl` — the only 0-3 grade data that exists
  today. The judged file holds real human/LLM-drafted grades (2s and 3s) but
  keyed to the *old* chunk_id scheme and only for a read subset per query
  (~5-14 chunks); the pool file has the real, current per-query candidate-set
  sizes (~22-34) but every chunk currently defaults to grade 0 or 1, because
  the judged grades no longer match any current chunk_id (see the id-scheme
  change discussion). This mode reassigns each query's real judged grade
  *values* onto that many of its current default-grade-1 ("target, unread")
  candidates, deterministically by query_id, so query-level candidate-set
  size and grade distribution both stay realistic. It is a proxy, not a
  measurement — good enough to reason about the shape of the tradeoff, not to
  make a final call.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import typer

from research_assistant.contracts.judge_J import JudgeMeta, RelevanceVerdict
from research_assistant.reranker.pairs_B import build_pairs

app = typer.Typer(add_completion=False, help="Sweep min_score_gap and report pair yield.")

REPO = Path(__file__).resolve().parent.parent
DRAFTS = REPO / "eval" / "datasets" / "drafts"
GAPS = (1, 2, 3)  # every gap possible on a 0-3 scale


def _simulated_verdicts() -> list[RelevanceVerdict]:
    judged = json.loads((DRAFTS / "llm_draft_grades_B.json").read_text(encoding="utf-8"))
    pool_lines = (DRAFTS / "pool_B.jsonl").read_text(encoding="utf-8").splitlines()
    pool = [json.loads(line) for line in pool_lines if line.strip()]

    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_query[row["query_id"]].append(row)

    meta = JudgeMeta(judge_model="simulated-from-drafts", prompt_version="sweep-v1")
    verdicts: list[RelevanceVerdict] = []
    for query_id, rows in by_query.items():
        rng = random.Random(query_id)  # deterministic per query
        # Real judged grade *values* for this query (order doesn't matter; only
        # the multiset of 1/2/3 grades a human/LLM actually assigned does).
        real_grades = list(judged.get(query_id, {}).values())
        candidates = [r for r in rows if r["draft_grade"] == 1]  # "target, unread" only
        rng.shuffle(candidates)
        upgrade_grades = real_grades[: len(candidates)]
        upgraded_ids = set()
        for row, grade in zip(candidates, upgrade_grades, strict=False):
            upgraded_ids.add(row["chunk_id"])
            verdicts.append(
                RelevanceVerdict(
                    query=query_id,
                    chunk_id=row["chunk_id"],
                    grade=grade,  # type: ignore[arg-type]
                    confidence=1.0,
                    rationale="simulated from llm_draft_grades_B.json",
                    meta=meta,
                )
            )
        for row in rows:
            if row["chunk_id"] in upgraded_ids:
                continue
            verdicts.append(
                RelevanceVerdict(
                    query=query_id,
                    chunk_id=row["chunk_id"],
                    grade=row["draft_grade"],  # type: ignore[arg-type]
                    confidence=1.0,
                    rationale=row["draft_reason"],
                    meta=meta,
                )
            )
    return verdicts


def _load_real(path: Path) -> list[RelevanceVerdict]:
    return [
        RelevanceVerdict.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


JUDGE_SCORES_OPTION = typer.Option(
    None, "--judge-scores", help="Real RelevanceVerdict JSONL. Omit to use the simulated proxy."
)
MAX_PAIRS_OPTION = typer.Option(4, help="Same cap build_pairs uses in production.")


@app.command()
def main(
    judge_scores: Path | None = JUDGE_SCORES_OPTION,
    max_pairs_per_query: int = MAX_PAIRS_OPTION,
) -> None:
    if judge_scores is not None:
        verdicts = _load_real(judge_scores)
        typer.echo(f"Using real judge scores: {judge_scores} ({len(verdicts)} verdicts)\n")
    else:
        verdicts = _simulated_verdicts()
        typer.secho(
            "No --judge-scores given: using a SIMULATED proxy (real judged grade values, "
            "redistributed onto the current candidate-pool structure). Treat these numbers "
            "as shape-of-the-tradeoff, not a final answer — rerun with --judge-scores once "
            "Sude's judge has scored the real pools.\n",
            fg=typer.colors.YELLOW,
        )
        n_sim_queries = len({v.query for v in verdicts})
        typer.echo(f"Simulated {len(verdicts)} verdicts across {n_sim_queries} queries\n")

    n_queries = len({v.query for v in verdicts})

    def lookup(chunk_id: str) -> str:
        return chunk_id  # only counting; real text is irrelevant to yield/coverage

    cols = (
        f"{'gap':>4}  {'pairs':>6}  {'% of gap=1':>10}  "
        f"{'queries w/ pairs':>17}  {'coverage':>9}"
    )
    typer.echo(cols)
    typer.echo("-" * len(cols))
    baseline: int | None = None
    for gap in GAPS:
        pairs = build_pairs(
            verdicts,
            chunk_lookup=lookup,
            min_score_gap=gap,
            max_pairs_per_query=max_pairs_per_query,
        )
        if baseline is None:
            baseline = len(pairs) or 1
        covered = len({p.query for p in pairs})
        pct = 100.0 * len(pairs) / baseline
        coverage = 100.0 * covered / n_queries
        row = (
            f"{gap:>4}  {len(pairs):>6}  {pct:>9.1f}%  "
            f"{covered:>10}/{n_queries:<5}  {coverage:>8.1f}%"
        )
        typer.echo(row)

    typer.echo(
        "\nThis table answers pair yield and query coverage only. Before locking in a gap:\n"
        "  1. Run Sude's real judge over the real candidate pools -> real RelevanceVerdict "
        "JSONL, rerun this script with --judge-scores against it.\n"
        "  2. On a subset with human qrels, compare the judge's chosen>rejected ordering "
        "against the human grades at each gap (agreement rate) -- a gap can look generous "
        "here and still be unreliable if the judge disagrees with humans at that margin.\n"
        "  3. Train one DPO run per candidate gap (same seed/epochs/lr, only min_score_gap "
        "differs) and compare nDCG@5/MRR/recall@20 on the held-out qrels -- the only thing "
        "that actually settles the choice."
    )


if __name__ == "__main__":
    app()
