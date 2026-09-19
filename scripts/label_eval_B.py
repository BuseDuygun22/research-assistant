"""Stage 05 review tool: promote reviewed drafts into the frozen eval set.

`scripts/build_eval_drafts_B.py` pools candidates and drafts a grade per
(query, chunk); a human reads them (`--view`) and marks a `pool_B.jsonl` row
reviewed by setting `reviewed: true` and `reviewer: "<name>"`, optionally
overriding `grade` when they disagree with the draft. This script is the only
path from those drafts into `eval/datasets/queries_B.jsonl` /
`qrels_B.jsonl` — the files `contracts.eval_dataset_J` calls "frozen":
changing a label there retroactively changes what every past gate run meant,
so this script never overwrites an existing (query_id, chunk_id) qrel's grade
silently. A changed grade needs `--allow-changed` and prints exactly what
changed and why it's allowed, so that decision is visible in the command that
made it rather than buried in a diff.

Only reviewed rows are promoted, and only for queries with at least one
reviewed row — an eval query with zero human-checked labels is not "held-out
and hand-labeled", it is still a draft, whatever the pool file says its
default grade is.

Usage:
  .venv/Scripts/python.exe scripts/label_eval_B.py                 # dry run: report only
  .venv/Scripts/python.exe scripts/label_eval_B.py --write         # promote new rows
  .venv/Scripts/python.exe scripts/label_eval_B.py --write --allow-changed
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import typer

from research_assistant.contracts.eval_dataset_J import EvalQuery, Qrel

app = typer.Typer(add_completion=False, help="Promote reviewed drafts into the frozen eval set.")

REPO = Path(__file__).resolve().parent.parent
DRAFTS = REPO / "eval" / "datasets" / "drafts"
QUERIES_DRAFT = DRAFTS / "queries_draft_B.jsonl"
POOL = DRAFTS / "pool_B.jsonl"
QUERIES_OUT = REPO / "eval" / "datasets" / "queries_B.jsonl"
QRELS_OUT = REPO / "eval" / "datasets" / "qrels_B.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _write_jsonl(rows: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")


def reviewed_qrels(pool_rows: list[dict[str, Any]]) -> list[Qrel]:
    """Build `Qrel`s from every reviewed pool row. `grade` overrides `draft_grade`
    when a reviewer set it; `reviewer` becomes `labeler` and is required -- a row
    with `reviewed: true` but no reviewer name did not actually get looked at."""
    qrels: list[Qrel] = []
    for row in pool_rows:
        if not row.get("reviewed"):
            continue
        reviewer = row.get("reviewer")
        if not reviewer:
            raise ValueError(
                f"{row['query_id']}/{row['chunk_id']} is marked reviewed but has no "
                "reviewer name; who reviewed this?"
            )
        grade = row["grade"] if row.get("grade") is not None else row["draft_grade"]
        qrels.append(
            Qrel(query_id=row["query_id"], chunk_id=row["chunk_id"], grade=grade, labeler=reviewer)
        )
    return qrels


@app.command()
def main(
    write: bool = typer.Option(False, "--write", help="Actually write the frozen files."),
    allow_changed: bool = typer.Option(
        False,
        "--allow-changed",
        help="Permit overwriting an existing (query_id, chunk_id) qrel's grade. Without "
        "this, a changed grade is reported and the run stops -- every metric computed "
        "against the old grade is otherwise silently invalidated with no record of why.",
    ),
) -> None:
    query_drafts = {row["query_id"]: row for row in _read_jsonl(QUERIES_DRAFT)}
    pool_rows = _read_jsonl(POOL)
    new_qrels = reviewed_qrels(pool_rows)

    reviewed_query_ids = {q.query_id for q in new_qrels}
    missing = reviewed_query_ids - set(query_drafts)
    if missing:
        raise ValueError(f"reviewed qrels reference unknown query_id(s): {sorted(missing)}")

    existing_qrels = {(q["query_id"], q["chunk_id"]): q["grade"] for q in _read_jsonl(QRELS_OUT)}
    existing_queries = {q["query_id"] for q in _read_jsonl(QUERIES_OUT)}

    added, changed, unchanged = [], [], []
    for q in new_qrels:
        key = (q.query_id, q.chunk_id)
        if key not in existing_qrels:
            added.append(q)
        elif existing_qrels[key] != q.grade:
            changed.append(q)
        else:
            unchanged.append(q)

    typer.echo(f"reviewed qrels: {len(new_qrels)} across {len(reviewed_query_ids)} queries")
    typer.echo(f"  new           : {len(added)}")
    typer.echo(f"  changed grade : {len(changed)}")
    typer.echo(f"  unchanged     : {len(unchanged)}")
    if changed:
        typer.secho("changed grades (old -> new):", fg=typer.colors.YELLOW)
        for q in changed:
            old = existing_qrels[(q.query_id, q.chunk_id)]
            typer.echo(f"  {q.query_id}/{q.chunk_id}: {old} -> {q.grade} (labeler={q.labeler})")
        if not allow_changed:
            typer.secho(
                "refusing to promote: use --allow-changed if this correction is intended.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    grades = Counter(q.grade for q in new_qrels)
    typer.echo(f"grade distribution: {dict(sorted(grades.items()))}")

    if not write:
        typer.echo("dry run: nothing written. Pass --write to promote.")
        return

    # Merge: existing qrels not touched by this run, plus every new/changed one.
    merged_qrels: dict[tuple[str, str], Qrel] = {
        (row["query_id"], row["chunk_id"]): Qrel(**row) for row in _read_jsonl(QRELS_OUT)
    }
    for q in new_qrels:
        merged_qrels[(q.query_id, q.chunk_id)] = q

    all_query_ids = existing_queries | reviewed_query_ids
    merged_queries = [
        EvalQuery(
            query_id=row["query_id"],
            query=row["query"],
            intent=row.get("intent", "factoid"),
            split=row.get("split", "test"),
            notes=row.get("notes"),
        )
        for qid, row in query_drafts.items()
        if qid in all_query_ids
    ]
    # Queries already frozen but no longer present in the current draft (a query
    # spec was removed) keep their frozen row rather than silently dropping it.
    drafted_ids = {q.query_id for q in merged_queries}
    for row in _read_jsonl(QUERIES_OUT):
        if row["query_id"] not in drafted_ids:
            merged_queries.append(EvalQuery(**row))

    _write_jsonl(merged_queries, QUERIES_OUT)
    _write_jsonl(list(merged_qrels.values()), QRELS_OUT)
    typer.secho(
        f"wrote {len(merged_queries)} queries, {len(merged_qrels)} qrels to "
        f"{QUERIES_OUT.relative_to(REPO)} / {QRELS_OUT.relative_to(REPO)}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
