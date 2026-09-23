"""Fetch a bounded arXiv corpus on fraud detection for this project.

Stage 00 companion script (see notebooks/00_corpus_scoping_B.ipynb for the sizing
and topic-scoping rationale). Unlike scripts/fetch_qasper_B.py, this produces real
PDFs, so it exercises the actual stage-01 PyMuPDF parsing path rather than
substituting pre-parsed text.

Scope: "fraud detection" restricted to categories where fraud is a machine-learning
/ security / quantitative-finance topic (cs.LG, cs.CR, cs.AI, stat.ML, q-fin.RM,
q-fin.ST). This excludes unrelated senses of "fraud" (legal, political, journalism)
that a bare keyword search would otherwise pull in, per stage 00's rule that the
domain boundary has to be written down and defended, not left to whatever a search
term happens to match.

The arXiv search/download mechanics live in `retrieval/arxiv_source_B.py`, shared
with the agent's opt-in live-discovery path (`RA_ALLOW_LIVE_DISCOVERY`) — this
script is the offline, human-run, "build the whole corpus once" use of the same
primitives; live discovery is the online, agent-run, "find a few more papers for
this one question" use of them.

Outputs:
  data/raw/<paper_id>.pdf              -- the actual PDF, parsed by stage 01
  data/corpus_manifest_B.jsonl         -- one row per paper (stage-00 manifest)

Does NOT write eval/datasets/*.jsonl. This corpus has no pre-existing evidence
annotations the way QASPER did, so queries_B.jsonl and qrels_B.jsonl for this
corpus have to be hand-built in stage 05, from scratch, against real retrieval
output -- there is no shortcut for a fresh domain.

Usage:
  .venv/Scripts/python.exe scripts/fetch_arxiv_fraud_B.py --n-papers 30
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import typer

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path plumbing
    sys.path.insert(0, str(SRC))

from research_assistant.retrieval.arxiv_source_B import (  # noqa: E402
    DEFAULT_CATEGORIES as CATEGORIES,
)
from research_assistant.retrieval.arxiv_source_B import (  # noqa: E402
    download_pdf,
)
from research_assistant.retrieval.arxiv_source_B import (  # noqa: E402
    search as search_arxiv,
)

app = typer.Typer(add_completion=False)

REPO = Path(__file__).resolve().parent.parent


@app.command()
def main(
    n_papers: int = typer.Option(30, help="Target corpus size"),
    dry_run: bool = typer.Option(False, help="Search and list candidates without downloading"),
) -> None:
    raw_dir = REPO / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"searching arXiv: 'fraud detection' restricted to {CATEGORIES}")
    papers = search_arxiv("%22fraud+detection%22", max_results=n_papers, categories=CATEGORIES)
    print(f"found {len(papers)} candidates")

    if dry_run:
        for p in papers:
            print(f"  {p.arxiv_id}  [{p.category}]  {p.title[:70]}")
        return

    manifest_rows = []
    for i, p in enumerate(papers, start=1):
        pid = p.paper_id
        dest = raw_dir / f"{pid}.pdf"
        print(f"[{i}/{len(papers)}] {p.arxiv_id}  {p.title[:60]}")
        try:
            download_pdf(p, dest)
        except Exception as exc:  # noqa: BLE001 -- log and continue, one bad PDF shouldn't kill the run
            print(f"    FAILED: {exc}")
            continue
        manifest_rows.append(
            {
                "paper_id": pid,
                "filename": f"{pid}.pdf",
                "title": p.title,
                "year": p.year,
                "venue": f"arXiv:{p.arxiv_id} ({p.category})",
                "why_included": f"matched 'fraud detection' in {p.category}; abstract: "
                f"{p.summary[:180]}...",
            }
        )
        time.sleep(3)  # arXiv download etiquette: no more than one request per 3 seconds

    (REPO / "data" / "corpus_manifest_B.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest_rows) + "\n",
        encoding="utf-8",
    )
    print(f"\ndownloaded {len(manifest_rows)}/{len(papers)} PDFs to {raw_dir}")
    print("wrote manifest: data/corpus_manifest_B.jsonl")
    if manifest_rows:
        by_cat: dict[str, int] = {}
        for r in manifest_rows:
            cat = r["venue"].split("(")[-1].rstrip(")")
            by_cat[cat] = by_cat.get(cat, 0) + 1
        print("\nby category:")
        for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
            print(f"  {cat:10s} {n}")
        years = [r["year"] for r in manifest_rows]
        print(f"\nyear range: {min(years)}-{max(years)}")


if __name__ == "__main__":
    app()
