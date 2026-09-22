"""The calibration CLI, end to end on a fixture corpus (Sude).

The point of this file: prove the tool actually catches a bad judge rather than
always reporting a reassuring number. `StubLLM.grade_chunk` always returns grade
3 (`llm_S.py:_stub_payload`) regardless of what is asked - a constant-output judge
is exactly the failure `cohens_kappa` exists to catch (design review E-something:
"a judge that is highly self-consistent and consistently wrong"). If this test
ever came back green with a HIGH kappa, the calibration instrument itself would be
broken, not the pipeline - that is what it is asserting against.
"""

from __future__ import annotations

import json

import pytest

from research_assistant.contracts.retrieval_J import Chunk, ChunkMetadata
from research_assistant.llm_S import set_llm
from research_assistant.mcp_server.backend_S import StubRetrieval, set_backend
from scripts.calibrate_judge_S import KAPPA_TRIPWIRE, load_labelled_pairs, main


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=text,
        metadata=ChunkMetadata(paper_id="p1", title="T", section="s", page=1),
    )


@pytest.fixture(autouse=True)
def reset():
    yield
    set_llm(None)
    set_backend(None)


def _write_dataset(tmp_path, n: int):
    """n queries, each with one chunk. Human grades cycle 0..3 so a
    constant-grade-3 judge disagrees on three quarters of them."""
    chunks = [_chunk(f"c{i:03d}", f"passage {i} about topic {i}") for i in range(n)]
    set_backend(StubRetrieval(chunks))
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.jsonl"
    with queries.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"query_id": f"q{i}", "query": f"question {i}"}) + "\n")
    with qrels.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(
                json.dumps({"query_id": f"q{i}", "chunk_id": f"c{i:03d}", "grade": i % 4})
                + "\n"
            )
    return queries, qrels


def test_a_constant_output_judge_fails_the_kappa_tripwire(tmp_path):
    """StubLLM (default judge_backend) always grades 3. Human grades cycle 0-3,
    so only a quarter agree - this must be reported as a bad judge, not a good one."""
    queries, qrels = _write_dataset(tmp_path, 40)
    code = main(["--qrels", str(qrels), "--queries", str(queries), "--min-labels", "30"])
    assert code == 1, "a constant-output judge must fail the calibration, not pass it"


def test_a_perfect_judge_clears_the_tripwire(tmp_path, monkeypatch):
    """A scripted judge that always echoes the human grade must clear kappa=1.0
    and exit 0 - the counterpart to the test above, so a passing run is trusted
    for the right reason and not because the script always returns 0."""
    queries, qrels = _write_dataset(tmp_path, 40)
    pairs = load_labelled_pairs(qrels, queries)
    human = [g for _, _, g in pairs]

    grades_iter = iter(human)

    class EchoJudge:
        """Echoes back whatever human grade the fixture expects next, in call
        order. Grading happens sequentially over `pairs` in `run_judge`, so this
        lines up 1:1 with the human-grade sequence read above."""

        model = "echo-test"

        def complete(self, system, user, max_tokens=1024):
            raise NotImplementedError

        def complete_json(self, system, user, schema, max_tokens=1024):
            # relevance_S._RelevanceReply: grade, confidence, rationale
            return schema(grade=next(grades_iter), confidence=0.95, rationale="echo")

    set_llm(EchoJudge())

    code = main(["--qrels", str(qrels), "--queries", str(queries), "--min-labels", "30"])
    assert code == 0


def test_refuses_below_the_label_floor(tmp_path):
    queries, qrels = _write_dataset(tmp_path, 5)
    code = main(["--qrels", str(qrels), "--queries", str(queries), "--min-labels", "30"])
    assert code == 2


def test_refuses_on_missing_files(tmp_path):
    code = main(
        ["--qrels", str(tmp_path / "nope.jsonl"), "--queries", str(tmp_path / "nope2.jsonl")]
    )
    assert code == 2


def test_unresolvable_chunk_ids_are_dropped_not_miscounted(tmp_path, capsys):
    """A stale chunk_id (wrong corpus_version) must not silently count as either
    agreement or disagreement - it should vanish from n and be reported on stderr."""
    set_backend(StubRetrieval([_chunk("real1", "text")]))
    queries = tmp_path / "q.jsonl"
    qrels = tmp_path / "qr.jsonl"
    queries.write_text(json.dumps({"query_id": "q1", "query": "x"}) + "\n", encoding="utf-8")
    qrels.write_text(
        json.dumps({"query_id": "q1", "chunk_id": "does-not-exist", "grade": 2}) + "\n",
        encoding="utf-8",
    )
    pairs = load_labelled_pairs(qrels, queries)
    assert pairs == []


def test_tripwire_matches_the_documented_value():
    """docs/architecture_J.md Stage 5: 'rework the rubric if kappa < 0.5.'"""
    assert KAPPA_TRIPWIRE == 0.5
