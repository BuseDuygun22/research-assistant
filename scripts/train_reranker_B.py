"""CLI for stage 08: DPO fine-tune the cross-encoder reranker.

Matches the Makefile target:

    $(PY) scripts/train_reranker_B.py --config configs/reranker_B.yaml

Needs the heavy extra (`make install-train`). Without it the command exits with a
single actionable message rather than an ImportError traceback. `--check` reports
readiness without training anything, and works on the base install.
"""

from __future__ import annotations

from pathlib import Path

import typer

from research_assistant.reranker.registry_B import load_reranker_config
from research_assistant.reranker.train_B import (
    TrainingDependenciesMissing,
    TrainSettings,
    build_dpo_rows,
    load_pairs,
    missing_dependencies,
    train,
    training_params,
)

app = typer.Typer(add_completion=False, help="DPO-train the cross-encoder reranker.")


# Module-level singletons: typer's idiom is a call in the default, which ruff's B008
# forbids inline.
CONFIG_OPTION = typer.Option(
    Path("configs/reranker_B.yaml"), "--config", "-c", help="Reranker config YAML."
)
RUN_NAME_OPTION = typer.Option("dpo_v1", "--run-name", help="MLflow run name.")
REGISTER_AS_OPTION = typer.Option(
    None,
    "--register-as",
    help="Register the result under this name. Registration never promotes; the gate does.",
)
CHECK_OPTION = typer.Option(
    False, "--check", help="Validate config, pairs and dependencies, then stop."
)


@app.command()
def main(
    config: Path = CONFIG_OPTION,
    run_name: str = RUN_NAME_OPTION,
    register_as: str | None = REGISTER_AS_OPTION,
    check: bool = CHECK_OPTION,
) -> None:
    """Train, log to MLflow, and optionally add a (non-champion) registry entry."""
    cfg = load_reranker_config(config)
    settings = TrainSettings.from_config(cfg)

    if check:
        pairs = load_pairs(settings.pairs_path)
        rows = build_dpo_rows(pairs)
        missing = missing_dependencies()
        typer.echo(f"config     : ok ({settings.base_model}, beta={settings.beta})")
        typer.echo(f"pairs      : {len(rows)} rows from {settings.pairs_path}")
        typer.echo(f"params     : {training_params(settings, len(pairs))}")
        if missing:
            typer.secho(
                f"deps       : MISSING {', '.join(missing)} -> "
                'run `pip install -e ".[train]"` before `make train`',
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=1)
        typer.secho("deps       : ok", fg=typer.colors.GREEN)
        return

    try:
        summary = train(cfg, run_name=run_name, register_as=register_as)
    except TrainingDependenciesMissing as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    for key, value in summary.items():
        typer.echo(f"{key}: {value}")
    typer.echo("champion unchanged; run the gate before promoting.")


if __name__ == "__main__":
    app()
