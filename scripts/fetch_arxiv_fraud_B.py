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

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import typer

app = typer.Typer(add_completion=False)

REPO = Path(__file__).resolve().parent.parent
ARXIV_API = "https://export.arxiv.org/api/query"
_HEADERS = {"User-Agent": "research-assistant-B/1.0 (stage-00 corpus fetch)"}
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# Scoped categories, not a bare keyword match. See module docstring.
CATEGORIES = ["cs.LG", "cs.CR", "cs.AI", "stat.ML", "q-fin.RM", "q-fin.ST"]


def _paper_id(arxiv_id: str) -> str:
    return hashlib.sha1(arxiv_id.encode()).hexdigest()[:10]


def _fetch_page(query: str, start: int, max_results: int) -> ET.Element:
    url = f"{ARXIV_API}?search_query={query}&start={start}&max_results={max_results}"
    # httpx rather than urllib: arXiv answers urllib's requests with HTTP 406 from
    # some networks (observed 2026-09-22) while accepting the same request from httpx.
    resp = httpx.get(url, headers=_HEADERS, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    return ET.fromstring(resp.content)


def _search(n_papers: int) -> list[dict]:
    cat_clause = "+OR+".join(f"cat:{c}" for c in CATEGORIES)
    query = f"%28all:%22fraud+detection%22%29+AND+%28{cat_clause}%29"

    papers, start, page_size = [], 0, 50
    seen_ids = set()
    while len(papers) < n_papers and start < 300:  # hard stop against a pathological loop
        root = _fetch_page(query, start, page_size)
        entries = root.findall("a:entry", NS)
        if not entries:
            break
        for e in entries:
            arxiv_id = e.find("a:id", NS).text.rsplit("/", 1)[-1]
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)  # strip version suffix
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            title = " ".join(e.find("a:title", NS).text.split())
            summary = " ".join(e.find("a:summary", NS).text.split())
            published = e.find("a:published", NS).text
            year = int(published[:4])
            pdf_link = None
            for link in e.findall("a:link", NS):
                if link.get("title") == "pdf":
                    pdf_link = link.get("href")
            if pdf_link is None:
                continue
            primary_cat = e.find("arxiv:primary_category", NS)
            category = primary_cat.get("term") if primary_cat is not None else ""
            papers.append(
                dict(
                    arxiv_id=arxiv_id,
                    title=title,
                    summary=summary,
                    year=year,
                    category=category,
                    pdf_url=pdf_link if pdf_link.endswith(".pdf") else pdf_link + ".pdf",
                )
            )
            if len(papers) >= n_papers:
                break
        start += page_size
        time.sleep(3)  # arXiv API etiquette: no more than one request per 3 seconds
    return papers


@app.command()
def main(
    n_papers: int = typer.Option(30, help="Target corpus size"),
    dry_run: bool = typer.Option(False, help="Search and list candidates without downloading"),
) -> None:
    raw_dir = REPO / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"searching arXiv: 'fraud detection' restricted to {CATEGORIES}")
    papers = _search(n_papers)
    print(f"found {len(papers)} candidates")

    if dry_run:
        for p in papers:
            print(f"  {p['arxiv_id']}  [{p['category']}]  {p['title'][:70]}")
        return

    manifest_rows = []
    for i, p in enumerate(papers, start=1):
        pid = _paper_id(p["arxiv_id"])
        dest = raw_dir / f"{pid}.pdf"
        print(f"[{i}/{len(papers)}] {p['arxiv_id']}  {p['title'][:60]}")
        try:
            resp = httpx.get(p["pdf_url"], headers=_HEADERS, timeout=60, follow_redirects=True)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        except Exception as exc:  # noqa: BLE001 -- log and continue, one bad PDF shouldn't kill the run
            print(f"    FAILED: {exc}")
            continue
        manifest_rows.append(
            {
                "paper_id": pid,
                "filename": f"{pid}.pdf",
                "title": p["title"],
                "year": p["year"],
                "venue": f"arXiv:{p['arxiv_id']} ({p['category']})",
                "why_included": f"matched 'fraud detection' in {p['category']}; abstract: "
                f"{p['summary'][:180]}...",
            }
        )
        time.sleep(3)  # arXiv download etiquette

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
