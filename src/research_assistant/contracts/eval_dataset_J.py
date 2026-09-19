"""Held-out eval dataset contracts (JOINT — Buse produces, Sude's CI consumes).

Buse's `eval/datasets/queries_B.jsonl` and `qrels_B.jsonl` are read through
these models and never modified downstream. If the CI gate needs a new field,
it is requested from Buse, not patched locally.

DRAFT — requires Buse's sign-off before either side builds against it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GRADE_MEANING: dict[int, str] = {
    3: "Directly answers the query; sufficient to support the answer on its own.",
    2: "Meaningfully helps answer the query, but is incomplete or requires "
    "additional context/evidence.",
    1: "Related to the query/topic but does not actually help answer the "
    "specific question.",
    0: "Not relevant to answering the query.",
}
"""What `Qrel.grade` measures, and only that: *how useful is this chunk for
answering this query?* Deliberately narrow -- correctness, completeness,
confidence, citation quality, faithfulness and writing quality are each a real
axis, but scoring them here would contaminate the one thing qrels are for.
Retrieval quality (nDCG/MRR, this file) is graded before generation happens at
all; whether the eventual answer is faithful or actually addresses the question
is graded downstream, on the draft, by Sude's `FaithfulnessVerdict` /
`AnswerVerdict` (`contracts/judge_J.py`) -- never by re-grading a chunk.

Full labeling instructions, with examples per grade: `docs/qrels_labeling_guide_B.md`.
"""


class EvalQuery(BaseModel):
    """One held-out question, traceable back to the domain brief."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    query: str
    intent: Literal["factoid", "comparison", "synthesis", "method", "unanswerable"] = "factoid"
    split: Literal["dev", "test"] = "test"
    notes: str | None = Field(None, description="Why this query is in the set.")


class Qrel(BaseModel):
    """A graded relevance label for one (query, chunk) pair.

    `chunk_id` must come from `Chunk.derive_id`, so labels survive a re-ingest of
    the same corpus version.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    chunk_id: str
    grade: Literal[0, 1, 2, 3]
    labeler: str


class EvalDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[EvalQuery]
    qrels: list[Qrel]
    corpus_version: str = Field(..., description="Corpus these labels were made against.")

    def relevant(self, query_id: str) -> dict[str, int]:
        return {q.chunk_id: q.grade for q in self.qrels if q.query_id == query_id and q.grade > 0}

    def for_split(self, split: str) -> list[EvalQuery]:
        return [q for q in self.queries if q.split == split]


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_eval_dataset(queries_path: Path, qrels_path: Path, corpus_version: str) -> EvalDataset:
    """Strict load: a malformed row fails the CI gate rather than silently
    shrinking the eval set, which would make the gate easier to pass."""
    return EvalDataset(
        queries=[EvalQuery(**row) for row in _read_jsonl(queries_path)],
        qrels=[Qrel(**row) for row in _read_jsonl(qrels_path)],
        corpus_version=corpus_version,
    )
