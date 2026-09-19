"""CLI for the Track A ingestion pipeline.

Thin on purpose: every decision lives in `ingestion/pipeline_B.py` and the configs, and
this file only chooses what to print. Stage reasoning is in notebooks 01 to 03.

    .venv/Scripts/python.exe scripts/ingest_B.py --dry-run
    .venv/Scripts/python.exe scripts/ingest_B.py --stage chunk --limit 3
    .venv/Scripts/python.exe scripts/ingest_B.py --stage all
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

# Running this as a script rather than via the installed package should still work: a
# teammate cloning the repo gets a working CLI before `pip install -e .`.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path plumbing
    sys.path.insert(0, str(SRC))

from research_assistant.config_J import load_config, resolve_path  # noqa: E402
from research_assistant.ingestion.chunk_B import count_tokens, get_encoder  # noqa: E402
from research_assistant.ingestion.pipeline_B import IngestionResult, run_ingestion  # noqa: E402

app = typer.Typer(add_completion=False, help="Parse, chunk, embed and index the paper corpus.")
console = Console()


class Stage(str, Enum):
    parse = "parse"
    chunk = "chunk"
    embed = "embed"
    all = "all"


def _summary_table(result: IngestionResult) -> Table:
    table = Table(title="Ingestion summary", header_style="bold")
    table.add_column("paper_id")
    table.add_column("file", overflow="fold")
    table.add_column("pages", justify="right")
    table.add_column("kept", justify="right")
    table.add_column("sections", justify="right")
    table.add_column("chunks", justify="right")
    table.add_column("body font", justify="right")
    for row in result.per_paper():
        table.add_row(
            str(row["paper_id"]),
            str(row["file"]),
            str(row["pages"]),
            str(row["pages_kept"]),
            str(row["sections"]),
            str(row["chunks"]),
            f"{row['body_font']:.1f}",
        )
    return table


def _token_table(stats: dict[str, float]) -> Table:
    table = Table(title="Chunk token distribution", header_style="bold")
    for key in ("count", "min", "p25", "median", "p75", "max", "mean"):
        table.add_column(key, justify="right")
    table.add_row(*[f"{stats.get(k, 0):g}" for k in ("count", "min", "p25", "median", "p75",
                                                     "max", "mean")])
    return table


def _print_report(result: IngestionResult, cfg: dict[str, Any]) -> None:
    report = result.parse_report
    mode = "DRY RUN (nothing written)" if result.dry_run else f"stage={result.stage}"
    console.print(f"[bold]Ingestion[/bold] — {mode}")
    console.print(_summary_table(result))

    console.print(
        f"papers={report.get('n_papers', 0)}  "
        f"pages={report.get('total_pages', 0)}  "
        f"pages_kept={report.get('pages_kept', 0)}  "
        f"records_kept={report.get('records_kept', 0)}  "
        f"records_dropped={report.get('records_dropped', 0)}"
    )

    thin = report.get("thin_pages") or []
    if thin:
        console.print(
            f"[yellow]{len(thin)} page(s) below "
            f"parse.min_chars_per_page={cfg['parse']['min_chars_per_page']} — "
            f"inspect these, they are a corpus problem, not a row to discard[/yellow]"
        )
        for row in thin[:10]:
            console.print(f"    {row['file']} p{row['page']}  {row['n_chars']} chars")

    if result.token_stats:
        console.print(_token_table(result.token_stats))
        c = cfg["chunk"]
        enc = get_encoder(c["tokenizer"])
        n_tokens = [count_tokens(k.text, enc) for k in result.chunks]
        low = [n for n in n_tokens if n < c["min_tokens"]]
        high = [n for n in n_tokens if n > c["max_tokens"]]
        verdict = "[green]within bounds[/green]" if not low and not high else "[red]OUT OF BOUNDS[/red]"
        console.print(
            f"bounds min_tokens={c['min_tokens']} max_tokens={c['max_tokens']}: {verdict}"
            f"  (under={len(low)} over={len(high)})"
        )

    if result.embedded:
        console.print(f"embedded {result.embedded} chunks, dim={result.embedding_dim}")
    if result.store_count is not None:
        console.print(f"vector store '{result.collection}' now holds {result.store_count} chunks")
    for path in result.artefacts:
        console.print(f"wrote {path}")


@app.command()
def main(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Override the ingestion config path. Defaults to "
                                      "configs/ingestion_B.yaml via config_J."),
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Only ingest the first N PDFs. For smoke runs.")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Parse and chunk but write nothing, load no model and "
                                       "touch no index."),
    ] = False,
    stage: Annotated[
        Stage, typer.Option("--stage", help="Run up to this stage. Stages are cumulative.")
    ] = Stage.all,
    keep_thin: Annotated[
        bool,
        typer.Option("--keep-thin", help="Keep pages below parse.min_chars_per_page instead of "
                                         "filtering them (they are always reported)."),
    ] = False,
    progress: Annotated[
        bool, typer.Option("--progress", help="Show the sentence-transformers progress bar.")
    ] = False,
) -> None:
    """Run the ingestion pipeline and print a summary."""
    if config is not None:
        # A caller-supplied path bypasses the config_J registry, so read it here and pass
        # the dict down rather than letting any module read YAML on its own.
        import yaml

        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    else:
        cfg = load_config("ingestion")

    raw_dir = resolve_path(cfg["corpus"]["raw_dir"])
    if not raw_dir.exists() or not any(raw_dir.glob("*.pdf")):
        console.print(
            f"[red]no PDFs in {raw_dir}[/red]\n"
            "Generate the synthetic fixture corpus first:\n"
            "  .venv/Scripts/python.exe tests/fixtures/make_synthetic_corpus_B.py"
        )
        raise typer.Exit(code=1)

    result = run_ingestion(
        config=cfg,
        stage=stage.value,
        limit=limit,
        dry_run=dry_run,
        keep_thin=keep_thin,
        show_progress=progress,
    )
    _print_report(result, cfg)


if __name__ == "__main__":
    app()
