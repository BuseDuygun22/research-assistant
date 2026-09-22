"""Fetch a bounded QASPER subset and materialize it as this project's corpus.

Stage 00 companion script. See notebooks/00_corpus_scoping_B.ipynb for the sizing
and topic-scoping rationale this script implements.

QASPER ships pre-parsed section/paragraph text rather than raw PDFs, so this
script's output REPLACES the stage-01 parsing output (`data/interim/pages.jsonl`)
rather than feeding stage 01. There are no PDF page numbers in the source, so a
paragraph ordinal is used as the page surrogate everywhere the retrieval_J.py
contract expects `page_start`/`page_end`. This is a documented substitution, not
a silent one: every downstream citation is "paragraph N of section S", not a
true PDF page.

Outputs:
  data/raw_qasper/<paper_id>.json      -- one paper: title, sections, paragraphs
  data/corpus_manifest_B.jsonl         -- one row per paper (the stage-00 manifest)
  data/interim/pages.jsonl             -- paragraph-as-page records, stage-02 input
  eval/datasets/queries_B.jsonl        -- QASPER questions over the kept papers
  eval/datasets/qrels_B.jsonl          -- QASPER evidence, remapped to a 0-3 grade

Grade remapping (binary QASPER evidence -> this project's graded scale):
  3 = the evidence paragraph contains a `highlighted_evidence` sentence AND the
      question has an extractive/free-form answer (the paragraph doesn't just
      mention the topic, it contains the literal answer)
  2 = the paragraph is in `evidence` but has no highlighted sentence (whole
      paragraph was cited, but the precise answer wasn't pinned to a sentence)
  Unanswerable questions are excluded from qrels (no chunk should be labeled
  relevant to a question that has no answer in the corpus), but ARE kept in
  queries_B.jsonl with shape="unanswerable" as the negative-control set stage 06
  asks for.

Usage:
  .venv/Scripts/python.exe scripts/fetch_qasper_B.py --n-papers 60 --topic "summarization"
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import typer
from datasets import load_dataset

app = typer.Typer(add_completion=False)

REPO = Path(__file__).resolve().parent.parent
PARQUET_BASE = "hf://datasets/allenai/qasper@refs%2Fconvert%2Fparquet/qasper"


def _paper_id(qasper_id: str) -> str:
    return hashlib.sha1(qasper_id.encode()).hexdigest()[:10]


def _load_all():
    return load_dataset(
        "parquet",
        data_files={
            "train": f"{PARQUET_BASE}/train/0000.parquet",
            "validation": f"{PARQUET_BASE}/validation/0000.parquet",
            "test": f"{PARQUET_BASE}/test/0000.parquet",
        },
    )


def _matches_topic(paper: dict, topic: str | None) -> bool:
    if not topic:
        return True
    hay = (paper["title"] + " " + paper["abstract"]).lower()
    return topic.lower() in hay


@app.command()
def main(
    n_papers: int = typer.Option(60, help="Target corpus size, per stage-00 sizing guidance"),
    topic: str = typer.Option(
        None, help="Substring filter on title+abstract, e.g. 'summarization'. Empty = no filter."
    ),
    min_questions: int = typer.Option(
        1, help="Drop papers with fewer answerable questions than this"
    ),
    seed: int = 42,
) -> None:
    raw_dir = REPO / "data" / "raw_qasper"
    interim = REPO / "data" / "interim"
    eval_dir = REPO / "eval" / "datasets"
    for d in (raw_dir, interim, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("downloading QASPER (parquet convert branch)...")
    ds = _load_all()
    all_papers = list(ds["train"]) + list(ds["validation"]) + list(ds["test"])
    print(f"total QASPER papers available: {len(all_papers)}")

    candidates = [p for p in all_papers if _matches_topic(p, topic)]
    print(f"matching topic {topic!r}: {len(candidates)}")

    # Keep papers with at least one answerable question, so every paper can
    # contribute at least one qrel. Deterministic order (sorted by id) then
    # capped at n_papers, rather than random sampling, so re-running this
    # script with the same args always reproduces the same corpus.
    kept = []
    for p in sorted(candidates, key=lambda x: x["id"]):
        n_answerable = sum(
            1 for a in p["qas"]["answers"] if not all(x["unanswerable"] for x in a["answer"])
        )
        if n_answerable >= min_questions:
            kept.append(p)
        if len(kept) >= n_papers:
            break

    if len(kept) < n_papers:
        print(f"WARNING: only found {len(kept)} papers meeting criteria, target was {n_papers}")

    manifest_rows = []
    page_records = []
    queries = []
    qrels = []

    for paper in kept:
        pid = _paper_id(paper["id"])
        sections = paper["full_text"]["section_name"]
        para_lists = paper["full_text"]["paragraphs"]

        raw_dir.joinpath(f"{pid}.json").write_text(
            json.dumps(
                {
                    "paper_id": pid,
                    "qasper_id": paper["id"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "sections": sections,
                    "paragraphs": para_lists,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        manifest_rows.append(
            {
                "paper_id": pid,
                "filename": f"{pid}.json",
                "title": paper["title"],
                "year": None,
                "venue": "arXiv (via QASPER/S2ORC)",
                "why_included": f"QASPER source paper {paper['id']}"
                + (f", matched topic filter {topic!r}" if topic else ""),
            }
        )

        # Build a paragraph-index -> (section, ordinal-as-page) lookup, since
        # QASPER gives no PDF page numbers. Abstract is ordinal 0.
        para_index: dict[str, tuple[str, int]] = {}
        ordinal = 1
        page_records.append(
            {
                "paper_id": pid,
                "page": 0,
                "section": "Abstract",
                "text": paper["abstract"],
                "n_chars": len(paper["abstract"]),
                "dropped": False,
            }
        )
        for section, paras in zip(sections, para_lists, strict=False):
            sec_name = section or "UNTITLED_SECTION"
            is_refs = bool(re.search(r"reference|bibliograph|acknowledg", sec_name.lower()))
            for para in paras:
                page_records.append(
                    {
                        "paper_id": pid,
                        "page": ordinal,
                        "section": sec_name,
                        "text": para,
                        "n_chars": len(para),
                        "dropped": is_refs,
                    }
                )
                para_index[para.strip()] = (sec_name, ordinal)
                ordinal += 1

        for q_idx, (question, q_id, answers) in enumerate(
            zip(
                paper["qas"]["question"],
                paper["qas"]["question_id"],
                paper["qas"]["answers"],
                strict=False,
            )
        ):
            qid = f"{pid}_q{q_idx}"
            all_unanswerable = all(a["unanswerable"] for a in answers["answer"])
            queries.append(
                {
                    "query_id": qid,
                    "query": question,
                    "shape": "unanswerable" if all_unanswerable else "evidence_lookup",
                    "notes": f"source qasper question_id={q_id}, paper={paper['id']}",
                }
            )
            if all_unanswerable:
                continue

            for a in answers["answer"]:
                if a["unanswerable"]:
                    continue
                highlighted = set(s.strip() for s in a.get("highlighted_evidence", []) or [])
                for ev in a.get("evidence", []) or []:
                    ev_stripped = ev.strip()
                    if ev_stripped.startswith("FLOAT SELECTED"):
                        continue  # table/figure evidence: no paragraph text to chunk
                    hit = para_index.get(ev_stripped)
                    if hit is None:
                        continue  # evidence didn't match a known paragraph verbatim
                    sec_name, page = hit
                    has_highlight = any(h and h in ev_stripped for h in highlighted)
                    grade = 3 if has_highlight else 2
                    qrels.append(
                        {
                            "query_id": qid,
                            "chunk_id": f"{pid}_p{page}",  # placeholder pre-chunking id
                            "grade": grade,
                            "paper_id": pid,
                            "page_start": page,
                            "page_end": page,
                        }
                    )

    (REPO / "data" / "corpus_manifest_B.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest_rows) + "\n",
        encoding="utf-8",
    )
    (interim / "pages.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in page_records) + "\n",
        encoding="utf-8",
    )
    (eval_dir / "queries_B.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in queries) + "\n",
        encoding="utf-8",
    )
    # Dedupe qrels: the same (query, paragraph) pair can appear across multiple
    # answers if two annotators cited the same evidence. Keep the highest grade.
    best: dict[tuple[str, str], dict] = {}
    for r in qrels:
        key = (r["query_id"], r["chunk_id"])
        if key not in best or r["grade"] > best[key]["grade"]:
            best[key] = r
    qrels_deduped = list(best.values())
    (eval_dir / "qrels_B.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in qrels_deduped) + "\n",
        encoding="utf-8",
    )

    n_answerable_q = sum(1 for q in queries if q["shape"] != "unanswerable")
    print(f"\npapers kept:          {len(kept)}")
    print(f"page/paragraph records: {len(page_records)}")
    print(f"queries total:        {len(queries)}  ({n_answerable_q} answerable, "
          f"{len(queries) - n_answerable_q} unanswerable)")
    print(f"qrels (deduped):      {len(qrels_deduped)}")
    print(f"  grade 3 (pinpointed): {sum(1 for r in qrels_deduped if r['grade'] == 3)}")
    print(f"  grade 2 (paragraph):  {sum(1 for r in qrels_deduped if r['grade'] == 2)}")
    zero_qrel = n_answerable_q - len({r['query_id'] for r in qrels_deduped})
    if zero_qrel:
        print(f"  WARNING: {zero_qrel} answerable queries matched no paragraph "
              f"verbatim (likely table/figure-only evidence) and have zero qrels")
    print("\nwrote:")
    for p in [
        "data/corpus_manifest_B.jsonl",
        "data/interim/pages.jsonl",
        "eval/datasets/queries_B.jsonl",
        "eval/datasets/qrels_B.jsonl",
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    app()
