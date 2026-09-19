"""CLI for stage 07: build DPO preference pairs from Sude's judge scores.

Matches the Makefile target:

    $(PY) scripts/build_pairs_B.py --config configs/reranker_B.yaml

The leakage guard runs before anything is written, so a leaked eval query makes
this command exit non-zero with nothing on disk.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer

from research_assistant.reranker.pairs_B import (
    EvalLeakageError,
    build_pairs_from_config,
    normalise_query,
)
from research_assistant.reranker.registry_B import load_reranker_config, resolve_path

app = typer.Typer(add_completion=False, help="Build (query, chosen, rejected) preference pairs.")

# Module-level singletons: typer's idiom is a call in the default, which ruff's B008
# forbids inline.
CONFIG_OPTION = typer.Option(
    Path("configs/reranker_B.yaml"), "--config", "-c", help="Reranker config YAML."
)
DRY_RUN_OPTION = typer.Option(
    False, "--dry-run", help="Build and report, but do not write the output file."
)
PREVIEW_OPTION = typer.Option(
    0, "--preview", help="Print this many pairs for the five-minute sanity read."
)
MIN_SCORE_GAP_OPTION = typer.Option(
    None,
    "--min-score-gap",
    help="Override configs/reranker_B.yaml's pairs.min_score_gap for this run, e.g. to "
    "sweep gap=1/2/3 without editing the file between runs. Writes to a gap-suffixed "
    "output path so sweep runs never overwrite each other.",
)


@app.command()
def main(
    config: Path = CONFIG_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    preview: int = PREVIEW_OPTION,
    min_score_gap: int | None = MIN_SCORE_GAP_OPTION,
) -> None:
    """Build preference pairs and write them to `pairs.out_path`."""
    cfg = load_reranker_config(config)
    if min_score_gap is not None:
        cfg = {**cfg, "pairs": {**cfg["pairs"], "min_score_gap": min_score_gap}}
        out = Path(cfg["pairs"]["out_path"])
        cfg["pairs"]["out_path"] = str(out.with_stem(f"{out.stem}_gap{min_score_gap}"))
    try:
        pairs = build_pairs_from_config(cfg, write=not dry_run)
    except EvalLeakageError as exc:
        typer.secho(f"LEAKAGE: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    n_queries = len({normalise_query(p.query) for p in pairs})
    gaps = Counter(int(p.margin) for p in pairs)
    typer.echo(f"{len(pairs)} pairs from {n_queries} queries")
    for gap in sorted(gaps):
        typer.echo(f"  score gap {gap}: {gaps[gap]}")

    for pair in pairs[:preview]:
        typer.echo(f"\nQ: {pair.query}")
        typer.echo(f"  CHOSEN  (margin {pair.margin:.0f}): {pair.chosen_text[:160]!r}")
        typer.echo(f"  REJECTED: {pair.rejected_text[:160]!r}")

    if dry_run:
        typer.echo("dry run: nothing written")
    else:
        typer.echo(f"wrote {resolve_path(cfg['pairs']['out_path'])}")

    if len(pairs) < 200:
        typer.secho(
            f"warning: only {len(pairs)} pairs; stage 08 rarely moves the metric below ~200.",
            fg=typer.colors.YELLOW,
        )


if __name__ == "__main__":
    app()
