"""The gate's query floor and command line (Sude).

Two integration faults between the tracks, each pinned here:

* The gate hard-coded a 50-query floor while Track A's thresholds file - the
  file that owns the eval set - says 40, and its own domain brief targets 40 to
  60 queries. A legitimate 45-query run would have been refused by a number nobody
  on Track A chose.
* `make gate` passed `--thresholds`, a flag the gate did not have, and ran the
  gate as a script, which its package-relative imports do not support.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from eval.run_gate_S import MIN_QUERIES, GateRefusal, RunResult, compare, main, query_floor


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "thresholds.yaml"
    p.write_text(text, encoding="utf-8")
    return p


# --- reading the floor -------------------------------------------------------


def test_the_floor_comes_from_the_thresholds_file(tmp_path):
    assert query_floor(write(tmp_path, "dataset:\n  min_queries: 40\n")) == 40


def test_a_missing_file_falls_back_to_the_default(tmp_path):
    assert query_floor(tmp_path / "nope.yaml") == MIN_QUERIES
    assert query_floor(None) == MIN_QUERIES


@pytest.mark.parametrize(
    "text",
    [
        "dataset:\n  min_queries: 0\n",  # a floor of zero would switch the check off
        "dataset:\n  min_queries: -5\n",
        "dataset:\n  min_queries: many\n",
        "dataset:\n  min_queries: true\n",  # bool is an int subclass; must not pass as 1
        "dataset:\n  min_queries: 12.5\n",
        "dataset: nothing\n",
        "just: [unbalanced\n",
        "",
    ],
)
def test_a_malformed_floor_cannot_switch_the_check_off(tmp_path, text):
    """A bad thresholds file falls back to the default. It must never disable the
    floor, and it must never crash the gate."""
    assert query_floor(write(tmp_path, text)) == MIN_QUERIES


# --- the floor is applied ----------------------------------------------------


def _run(label: str, n: int, seed: int, shift: float = 0.0) -> RunResult:
    rng = random.Random(seed)
    r = RunResult(label=label, corpus_version="v", embedding_model="m", reranker_version="r")
    for i in range(n):
        base = rng.gauss(0.6, 0.1)
        r.per_query[f"q{i}"] = {"ndcg@5": min(1.0, base + shift)}
    return r


def test_a_run_above_track_as_floor_but_below_the_default_is_allowed():
    """45 queries: refused by the old hard-coded 50, fine under Track A's 40."""
    base, cand = _run("b", 45, seed=1), _run("c", 45, seed=1, shift=0.08)
    with pytest.raises(GateRefusal):
        compare(base, cand, min_effect=0.02, seed=0, n_resamples=200)
    verdict = compare(base, cand, min_effect=0.02, seed=0, n_resamples=200, min_queries=40)
    assert verdict.promote is True


def test_a_run_below_the_floor_is_still_refused_and_names_it():
    base, cand = _run("b", 30, seed=2), _run("c", 30, seed=2, shift=0.08)
    with pytest.raises(GateRefusal, match="40-query floor"):
        compare(base, cand, min_effect=0.02, seed=0, n_resamples=200, min_queries=40)


# --- the command line --------------------------------------------------------


def test_the_gate_accepts_the_flag_make_passes(tmp_path):
    """`make gate` passes --thresholds. An unknown flag is an argparse SystemExit(2),
    indistinguishable in CI from the gate's own 'refused' exit code."""
    thresholds = write(tmp_path, "dataset:\n  min_queries: 40\n")
    try:
        code = main(["--thresholds", str(thresholds), "--allow-degraded"])
    except SystemExit as exc:  # pragma: no cover - only on an unrecognised flag
        pytest.fail(f"gate rejected --thresholds (argparse exit {exc.code})")
    assert code in (0, 2)


def test_make_gate_runs_the_gate_as_a_module():
    """Run as a script the gate's relative imports fail with 'no known parent
    package'. It must be invoked with -m."""
    makefile = (Path(__file__).resolve().parents[3] / "Makefile").read_text(encoding="utf-8")
    gate_lines = [ln for ln in makefile.splitlines() if "run_gate_S" in ln]
    assert gate_lines, "Makefile has no gate target"
    assert all("-m eval.run_gate_S" in ln for ln in gate_lines), gate_lines
