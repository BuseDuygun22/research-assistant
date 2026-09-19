"""Held-out discipline, enforced mechanically (JOINT).

`architecture_J.md` says the `test` split never touches prompt or reranker
selection. That is a claim about behaviour, and the design says explicitly it is
"enforced mechanically rather than by good intentions" — this file is the
mechanism. Until it existed, the discipline was a sentence in a document.

Three separate leaks are checked, because they fail in different ways:

* **Split leakage** — the same query in both splits, so a "held-out" score is
  partly a training score.
* **Label leakage into training data** — a preference pair built from a test
  query, which trains the reranker on the set that is supposed to grade it.
* **Selection leakage** — code that reads the test split while choosing a prompt,
  a threshold, or a model. The subtlest of the three and the easiest to do by
  accident.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from research_assistant.config_J import EVAL_DIR, REPO_ROOT
from research_assistant.contracts.eval_dataset_J import EvalDataset, load_eval_dataset

QUERIES = EVAL_DIR / "datasets" / "queries_B.jsonl"
QRELS = EVAL_DIR / "datasets" / "qrels_B.jsonl"


def dataset() -> EvalDataset:
    if not QUERIES.exists() or QUERIES.stat().st_size == 0:
        pytest.skip("eval dataset not yet produced by Buse")
    return load_eval_dataset(QUERIES, QRELS, "test-corpus")


# --- split hygiene -----------------------------------------------------------


def test_no_query_appears_in_both_splits():
    ds = dataset()
    dev = {q.query_id for q in ds.for_split("dev")}
    test = {q.query_id for q in ds.for_split("test")}
    assert not (dev & test), f"query ids in both splits: {sorted(dev & test)}"


def test_no_duplicate_query_text_across_splits():
    """Different ids, same question, is the same leak wearing a hat."""
    ds = dataset()
    dev = {q.query.strip().lower() for q in ds.for_split("dev")}
    test = {q.query.strip().lower() for q in ds.for_split("test")}
    overlap = dev & test
    assert not overlap, f"same question text in both splits: {sorted(overlap)[:3]}"


def test_every_qrel_belongs_to_a_known_query():
    """A qrel for a query that does not exist means the two files have drifted,
    and a silently-dropped label makes the gate easier to pass."""
    ds = dataset()
    known = {q.query_id for q in ds.queries}
    orphans = {r.query_id for r in ds.qrels} - known
    assert not orphans, f"qrels reference unknown queries: {sorted(orphans)[:5]}"


# --- training-data hygiene ---------------------------------------------------


def test_preference_pairs_exclude_test_queries():
    """Pairs are what trains the reranker. A pair built from a test query trains
    the system on the set that is supposed to grade it."""
    ds = dataset()
    test_queries = {q.query.strip().lower() for q in ds.for_split("test")}
    pairs_path = REPO_ROOT / "data" / "preference_pairs.jsonl"
    if not pairs_path.exists() or pairs_path.stat().st_size == 0:
        pytest.skip("no preference pairs generated yet")

    import json

    leaked = [
        json.loads(line)["query"]
        for line in pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("query", "").strip().lower() in test_queries
    ]
    assert not leaked, f"preference pairs built from test queries: {leaked[:3]}"


# --- selection hygiene -------------------------------------------------------

SELECTION_MODULES = [
    "src/research_assistant/agents",
    "src/research_assistant/judge",
    "src/research_assistant/mcp_server",
]


def _string_constants(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
        return []
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


@pytest.mark.parametrize("module_dir", SELECTION_MODULES)
def test_runtime_code_does_not_read_the_test_split(module_dir):
    """Nothing in the serving path may name the test split.

    The gate may read it — that is its job. The agent graph, the judge and the
    tool layer may not: any of them consulting it while a human tunes a prompt is
    selection on the held-out set, which is the leak that does not show up as a
    duplicate row anywhere.
    """
    offenders = []
    for path in (REPO_ROOT / module_dir).rglob("*.py"):
        for value in _string_constants(path):
            if 'split="test"' in value or "qrels_B" in value or "queries_B" in value:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {value[:60]}")
    assert not offenders, "runtime code references the eval dataset:\n" + "\n".join(offenders)


def test_gate_warns_when_the_test_split_is_used_without_a_baseline():
    """The test split reports a final number; it does not choose between options.
    The gate says so out loud rather than relying on the reader to remember."""
    source = (REPO_ROOT / "eval" / "run_gate_S.py").read_text(encoding="utf-8")
    assert "must not be used to select" in source
