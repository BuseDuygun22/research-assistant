"""Retrieval metrics and the evaluation harness for Track A (Buse).

Promoted from `notebooks/06_metrics_baseline_B.ipynb`, which is the referee for
stages 01 through 04: nothing in the tactics ledger moves to `adopted` without a
number produced here. Sude's `eval/run_gate_S.py` imports this module directly, so
the public signatures below are a cross-track contract — see "Stable API" at the
bottom of this docstring before renaming anything.

Why four metrics and not one
----------------------------
A single headline number cannot tell you *which* stage to go fix, and the two big
failure modes have opposite fixes. The four are tracked together on purpose:

``nDCG@5`` — *Are the best chunks at the top of what the tools actually return?*
    The headline. Grade-aware (a 3 outranks a 2 outranks a 1) and position-aware
    (rank 1 is worth more than rank 5), so it is the metric a reranker moves.
    It uses the exponential-gain formulation, ``(2**grade - 1)``, which makes the
    gap between a grade-3 "answers the question outright" chunk and a grade-1
    "right paper, wrong section" chunk large rather than linear. That is deliberate:
    on the 0-3 scale of `contracts/eval_dataset_J.py` the grades are not evenly
    spaced in usefulness, and linear gain would let three near-misses outscore one
    correct answer.

``MRR`` — *How far down is the first genuinely good hit?*
    Catches the single worst user-visible failure: a correct chunk buried at rank 5
    that the reader never scrolls to. nDCG@5 dampens that; the reciprocal rank does
    not. Uses a ``min_grade`` cut (default 2) because "right topic, no answer" is
    not a hit.

``recall@20`` — *Did the candidate set contain the answer at all?*
    **This is a ceiling, not a score.** The reranker in stage 08 can only reorder the
    candidate set handed to it by fusion; it cannot conjure a chunk that hybrid
    search never retrieved. So recall@20 caps every downstream metric. If recall@20
    is low, tuning the ranker is wasted effort and the fix is upstream, in chunking
    (stage 02) or fusion (stage 04). This is also why the threshold file sets the
    highest floor (0.85) on this metric and gives it no regression tolerance: a
    ranking change has no business moving the ceiling at all.

``citation precision`` — *Do the page ranges we show actually support the claim?*
    The only metric that connects retrieval to what the reader sees. It is measured
    over the chunks that are actually cited, so a system that returns three results
    is judged on three, not penalised against a nominal k it never filled.

Why nDCG alone is not enough
----------------------------
nDCG@5 collapses two different diseases into one low number:

- **recall failure** (``recall@20 == 0``): the answer never entered the candidate
  set. nDCG@5 is 0. Fix in stages 02-04.
- **ranking failure** (``recall@20 > 0`` but ``nDCG@5`` low): the answer was in the
  candidate set and the ordering buried it. nDCG@5 is also low. Fix in stage 08.

Reading only nDCG you cannot tell them apart, and the ratio between them is exactly
what decides whether the DPO reranker is worth building for this corpus.
`evaluate` therefore returns a per-query DataFrame, not just the means, and
`failure_breakdown` splits it for you.

Undefined values: NaN, never 0.0
--------------------------------
The notebook prototype returned ``0.0`` when a query had no relevance labels. That
is a silent scoring bug: an unjudged query is indistinguishable from a query the
retriever failed, and every unjudged query drags the mean down uniformly, which
makes the baseline look worse than it is and the gate stricter than intended.

The rule here is:

- **Not scoreable for lack of labels** -> ``NaN``, and the query is excluded from
  that metric's mean (``n`` in the summary records how many queries actually
  counted). Applies when the query has no qrels at all, and for recall/MRR when it
  has qrels but none at or above ``min_grade``.
- **Not scoreable for lack of results** -> ``0.0``. An empty result list on a query
  that *does* have a correct answer is a retrieval failure and must be scored as
  one; returning NaN there would let a retriever dodge the gate by returning
  nothing.

Stable API (Sude's gate runner imports these; do not rename without telling her)
--------------------------------------------------------------------------------
``ndcg_at_k(ranked_ids, rel, k=5) -> float``
``mrr(ranked_ids, rel, min_grade=2) -> float``
``recall_at_k(ranked_ids, rel, k=20, min_grade=2) -> float``
``citation_precision(ranked_ids, rel, k=5, min_grade=2) -> float``
``load_eval_set(queries_path=None, qrels_path=None, *, min_queries=None) -> EvalSet``
``evaluate(search_fn, k=5, candidate_k=20, label="", eval_set=None) -> (DataFrame, dict)``
``compare(baseline_summary, candidate_summary, thresholds) -> CompareVerdict``
``log_to_mlflow(summary, params=None, ...) -> str | None``

Summary dictionaries are keyed by the *threshold file's* metric names
(``ndcg_at_5``, ``mrr``, ``recall_at_20``, ``citation_precision``), so
`compare` is a direct lookup against `eval/thresholds_B.yaml` with no name
mapping in between. The notebook's short column names (``ndcg5``, ``cite_prec``)
were dropped for that reason.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from research_assistant.contracts.eval_dataset_J import EvalQuery, Qrel

__all__ = [
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_K",
    "DEFAULT_MIN_GRADE",
    "CompareVerdict",
    "EvalSet",
    "EvalSetError",
    "MetricCheck",
    "citation_precision",
    "compare",
    "evaluate",
    "failure_breakdown",
    "load_eval_set",
    "load_thresholds",
    "log_to_mlflow",
    "metric_names",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
]

#: Default top-k shown to the user. Mirrors ``rerank.top_k`` in configs/retrieval_B.yaml.
DEFAULT_K = 5
#: Default fused candidate-set size. Mirrors ``hybrid.candidate_k``.
DEFAULT_CANDIDATE_K = 20
#: Grade at or above which a chunk counts as a genuine hit. Grade 2 = "partial answer";
#: grade 1 = "right paper, does not answer", which is not a hit. See GRADE_MEANING.
DEFAULT_MIN_GRADE = 2

NAN = float("nan")

# Grades map -> metric key. Kept in one place so evaluate(), compare() and the MLflow
# helper cannot drift apart.
_NDCG = "ndcg_at_{k}"
_MRR = "mrr"
_RECALL = "recall_at_{k}"
_CITE = "citation_precision"


# --------------------------------------------------------------------------------------
# repo paths
#
# NOTE (2026-09-14, Buse): config_J.REPO_ROOT was `parents[2].parent` — one level *above*
# the checkout — so `load_config` and `resolve_path` pointed outside the repo and every
# lookup raised FileNotFoundError. It has since been corrected to `parents[2]`. The
# fallback below is kept because config_J is a joint file that either track can move: the
# joint loader is tried first and a locally computed root is used only if it fails, so
# this module (and therefore Sude's gate) cannot be broken by a path change upstream.
# --------------------------------------------------------------------------------------

_LOCAL_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(rel: str | Path) -> Path:
    """Resolve a repo-relative config path, tolerating the config_J root bug."""
    p = Path(rel)
    if p.is_absolute():
        return p
    try:
        from research_assistant.config_J import resolve_path

        resolved = resolve_path(p)
        if resolved.exists():
            return resolved
    except Exception:  # pragma: no cover - joint module import is not this module's job
        pass
    return (_LOCAL_REPO_ROOT / p).resolve()


def load_thresholds() -> dict[str, Any]:
    """Load `eval/thresholds_B.yaml`, the single source of truth for the gate.

    Prefers ``config_J.load_config("thresholds")`` so there is one loader in the repo;
    falls back to reading the file directly while the joint loader's root is broken.
    """
    try:
        from research_assistant.config_J import load_config

        return load_config("thresholds")
    except Exception:
        import yaml

        path = _LOCAL_REPO_ROOT / "eval" / "thresholds_B.yaml"
        if not path.exists():
            raise FileNotFoundError(f"thresholds file not found at {path}") from None
        warnings.warn(
            "config_J.load_config('thresholds') failed (REPO_ROOT points outside the "
            f"repo); read {path} directly instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}


def metric_names(k: int = DEFAULT_K, candidate_k: int = DEFAULT_CANDIDATE_K) -> dict[str, str]:
    """Map short metric role -> the key used in summaries, MLflow and thresholds_B.yaml.

    Roles are ``ndcg``, ``mrr``, ``recall``, ``citation``. With the defaults this yields
    ``ndcg_at_5``, ``mrr``, ``recall_at_20``, ``citation_precision`` — exactly the keys
    under ``retrieval:`` in the threshold file.
    """
    return {
        "ndcg": _NDCG.format(k=k),
        "mrr": _MRR,
        "recall": _RECALL.format(k=candidate_k),
        "citation": _CITE,
    }


# --------------------------------------------------------------------------------------
# the four metrics
# --------------------------------------------------------------------------------------


def _dcg(grades: Iterable[int]) -> float:
    """Discounted cumulative gain with exponential gain, ``(2**g - 1) / log2(rank + 1)``."""
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked_ids: Sequence[str], rel: Mapping[str, int], k: int = DEFAULT_K) -> float:
    """Normalised discounted cumulative gain over the top ``k`` returned chunks.

    Answers: *are the best chunks at the top of what the tools return?*

    Grade-aware and position-aware, which is what makes it the headline metric and the
    one a reranker is trained to move. Uses the standard exponential-gain formulation
    ``(2**grade - 1)``, so a single grade-3 chunk (gain 7) outweighs two grade-1 chunks
    (gain 1 each) — the 0-3 scale is not linear in usefulness and the metric should not
    pretend it is. The ideal DCG is computed from the query's *whole* label set, capped
    at ``k``, so the score is "how close to the best possible top-``k``", not "how close
    to the best possible ordering of what you happened to return".

    Args:
        ranked_ids: Chunk ids in the order the retriever returned them, best first.
        rel: ``chunk_id -> grade`` for this query. Unlisted chunks count as grade 0.
        k: Cutoff. Defaults to the 5 results the MCP tools actually surface.

    Returns:
        A float in ``[0.0, 1.0]``, or ``NaN`` when the query has no labels with a
        non-zero grade — an unjudged query is not a score of zero, and counting it as
        one would silently drag the baseline down. See the module docstring.

    Note:
        Blind spot: nDCG@5 is low both when the candidate set never contained the answer
        and when it did and the ranking buried it. Always read it beside ``recall_at_k``.
    """
    ideal_grades = sorted((g for g in rel.values() if g > 0), reverse=True)[:k]
    denom = _dcg(ideal_grades)
    if denom <= 0.0:
        return NAN
    gains = [rel.get(cid, 0) for cid in ranked_ids[:k]]
    return _dcg(gains) / denom


def mrr(
    ranked_ids: Sequence[str],
    rel: Mapping[str, int],
    min_grade: int = DEFAULT_MIN_GRADE,
) -> float:
    """Reciprocal rank of the first chunk at or above ``min_grade``.

    Answers: *how far down is the first genuinely useful hit?*

    Tracked because it is brutally sensitive to the failure mode nDCG@5 smooths over:
    a correct chunk sitting at rank 5 that nobody scrolls to. First hit at rank 1 scores
    1.0, rank 2 scores 0.5, rank 4 scores 0.25.

    Computed over the **whole** ranked list, not a top-``k`` slice, so the metric can
    still distinguish "buried at rank 12" from "not found at all".

    Args:
        ranked_ids: Chunk ids, best first.
        rel: ``chunk_id -> grade`` for this query.
        min_grade: The hit threshold. 2 by default: grade 2 is "partial answer", grade 1
            is "right paper, does not answer the question", which is not a hit.

    Returns:
        ``1 / rank`` of the first hit; ``0.0`` if the query has hits available but none
        were returned (a real failure); ``NaN`` if the query has no chunk at or above
        ``min_grade`` anywhere in its labels, in which case there is nothing to find and
        the query is excluded from the mean.
    """
    if not any(g >= min_grade for g in rel.values()):
        return NAN
    for i, cid in enumerate(ranked_ids, start=1):
        if rel.get(cid, 0) >= min_grade:
            return 1.0 / i
    return 0.0


def recall_at_k(
    ranked_ids: Sequence[str],
    rel: Mapping[str, int],
    k: int = DEFAULT_CANDIDATE_K,
    min_grade: int = DEFAULT_MIN_GRADE,
) -> float:
    """Fraction of the query's known-good chunks that made it into the top ``k``.

    Answers: *did the candidate set contain the answer at all?*

    This is the **ceiling on everything downstream**. The reranker reorders the fused
    candidate set; it cannot retrieve. Whatever recall@20 misses is permanently lost to
    every metric measured after it, which is why the gate floors it highest (0.85) and
    gives it no regression tolerance. A low recall@20 means stop tuning the ranker and
    go back to chunking (stage 02) or fusion (stage 04).

    ``k`` defaults to ``candidate_k``, not ``top_k``: the question is about the set
    handed to the reranker, not the five results shown.

    Args:
        ranked_ids: Chunk ids, best first.
        rel: ``chunk_id -> grade`` for this query.
        k: Cutoff, normally the fused candidate-set size.
        min_grade: Grade at or above which a label counts as recallable.

    Returns:
        A float in ``[0.0, 1.0]``, or ``NaN`` when the query has no labelled chunk at or
        above ``min_grade``. NaN rather than 0.0 or 1.0 is the documented choice: recall
        of an empty target set is undefined, 0.0 would punish the retriever for a gap in
        the *labels*, and 1.0 would flatter it. The query is dropped from this metric's
        mean and ``n`` in the summary records that it was.
    """
    good = {cid for cid, g in rel.items() if g >= min_grade}
    if not good:
        return NAN
    return len(good & set(ranked_ids[:k])) / len(good)


def citation_precision(
    ranked_ids: Sequence[str],
    rel: Mapping[str, int],
    k: int = DEFAULT_K,
    min_grade: int = DEFAULT_MIN_GRADE,
) -> float:
    """Fraction of the chunks we would actually cite that genuinely support an answer.

    Answers: *do the page ranges the reader is shown really back the claim?*

    The only metric that reaches the user-visible surface — these are the chunks whose
    ``page_start``/``page_end`` end up in the answer's citations, and the ones Sude's
    editor agent argues with when it cannot ground a sentence.

    **Denominator is what was actually returned, not ``k``.** If the retriever returns 3
    results for a top-5 request, this is precision over those 3. Rationale: citing three
    good chunks is not 60% correct citation behaviour, it is correct citation behaviour
    on a short list. The cost of that choice, and it is a real one: a retriever that
    returns a single lucky hit scores 1.0. That is why this metric is never read alone —
    ``recall_at_k`` is what stops a one-result system from looking perfect, and the gate
    requires both.

    Args:
        ranked_ids: Chunk ids, best first.
        rel: ``chunk_id -> grade`` for this query.
        k: How many results would be cited.
        min_grade: Grade at or above which a citation is considered supported.

    Returns:
        A float in ``[0.0, 1.0]``; ``0.0`` when the result list is empty but the query
        has a findable answer (returning nothing is a failure, not an exemption); or
        ``NaN`` when the query has no labelled chunk at or above ``min_grade``, since
        then no returned chunk could ever have been counted as supported.
    """
    if not any(g >= min_grade for g in rel.values()):
        return NAN
    top = list(ranked_ids[:k])
    if not top:
        return 0.0
    return sum(1 for cid in top if rel.get(cid, 0) >= min_grade) / len(top)


# --------------------------------------------------------------------------------------
# dataset loading
# --------------------------------------------------------------------------------------


class EvalSetError(RuntimeError):
    """A held-out set failed to load or validate.

    Raised instead of skipping the offending row. A silently dropped qrel changes every
    metric downstream of it and nothing in the run would say so; the gate is allowed to
    refuse to run, it is not allowed to run on a quietly truncated dataset.
    """


@dataclass(frozen=True)
class EvalSet:
    """A validated held-out set: the queries, the judgments, and a lookup for scoring."""

    queries: tuple[EvalQuery, ...]
    qrels: tuple[Qrel, ...]
    queries_path: Path | None = None
    qrels_path: Path | None = None

    @property
    def rel_by_query(self) -> dict[str, dict[str, int]]:
        """``query_id -> {chunk_id: grade}``, the shape every metric function takes.

        A plain dict, never a defaultdict: the notebook used a defaultdict, so a typo in
        a query_id silently produced an empty label set and a score of 0.0 for a query
        that was in fact fully labelled. Here a missing key is a missing key.
        """
        out: dict[str, dict[str, int]] = {q.query_id: {} for q in self.queries}
        for r in self.qrels:
            out.setdefault(r.query_id, {})[r.chunk_id] = r.grade
        return out

    def __len__(self) -> int:
        return len(self.queries)


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.exists():
        raise EvalSetError(f"eval file not found: {path}")
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalSetError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise EvalSetError(f"{path}:{lineno} is a {type(obj).__name__}, expected an object")
            rows.append((lineno, obj))
    return rows


def load_eval_set(
    queries_path: str | Path | None = None,
    qrels_path: str | Path | None = None,
    *,
    min_queries: int | None = None,
    require_positive_label: bool = True,
) -> EvalSet:
    """Load and validate the held-out set against the joint contract models.

    Fails loudly on anything malformed — a bad line raises `EvalSetError` naming the file
    and line number rather than being skipped. Skipping is the dangerous behaviour here:
    the resulting numbers still look plausible, and nothing downstream can tell that the
    set shrank.

    Args:
        queries_path: Defaults to ``dataset.queries`` in `eval/thresholds_B.yaml`.
        qrels_path: Defaults to ``dataset.qrels`` there.
        min_queries: Refuse to return a set smaller than this. Defaults to
            ``dataset.min_queries`` (40) when the paths also came from the config; pass
            ``0`` to disable, which is what the example dataset and the tests do.
        require_positive_label: Enforce notebook 05's exit check that every query has at
            least one chunk graded above 0. A query with no correct answer in the corpus
            measures nothing and drags every metric down uniformly.

    Returns:
        An `EvalSet`.

    Raises:
        EvalSetError: on a missing file, malformed JSON, a schema violation, an orphan
            qrel, a duplicate id, a query with no positive label, or a set below
            ``min_queries``.
    """
    from_config = queries_path is None and qrels_path is None
    cfg_dataset: Mapping[str, Any] = {}
    if from_config or min_queries is None:
        try:
            cfg_dataset = load_thresholds().get("dataset", {}) or {}
        except FileNotFoundError:
            cfg_dataset = {}

    qpath = _resolve(queries_path or cfg_dataset.get("queries", "eval/datasets/queries_B.jsonl"))
    rpath = _resolve(qrels_path or cfg_dataset.get("qrels", "eval/datasets/qrels_B.jsonl"))
    if min_queries is None:
        min_queries = int(cfg_dataset.get("min_queries", 0)) if from_config else 0

    queries: list[EvalQuery] = []
    seen_qids: set[str] = set()
    for lineno, obj in _read_jsonl(qpath):
        try:
            q = EvalQuery(**obj)
        except ValidationError as exc:
            raise EvalSetError(f"{qpath}:{lineno} is not a valid EvalQuery:\n{exc}") from exc
        if q.query_id in seen_qids:
            raise EvalSetError(f"{qpath}:{lineno} duplicate query_id {q.query_id!r}")
        seen_qids.add(q.query_id)
        queries.append(q)

    qrels: list[Qrel] = []
    seen_pairs: set[tuple[str, str]] = set()
    for lineno, obj in _read_jsonl(rpath):
        try:
            r = Qrel(**obj)
        except ValidationError as exc:
            raise EvalSetError(f"{rpath}:{lineno} is not a valid Qrel:\n{exc}") from exc
        if r.query_id not in seen_qids:
            raise EvalSetError(
                f"{rpath}:{lineno} judges unknown query_id {r.query_id!r}; "
                f"an orphan qrel means the two files are out of sync"
            )
        # The merged `Qrel` contract (Sude's) carries no page range -- that provenance
        # now lives on `Chunk.metadata.page` (singular) instead, so the page_end >=
        # page_start guard this loader used to enforce has nothing left to check here.
        key = (r.query_id, r.chunk_id)
        if key in seen_pairs:
            raise EvalSetError(f"{rpath}:{lineno} duplicate judgment for {key}")
        seen_pairs.add(key)
        qrels.append(r)

    if min_queries and len(queries) < min_queries:
        raise EvalSetError(
            f"{qpath} has {len(queries)} queries, below the configured minimum of "
            f"{min_queries}; nDCG is not stable on a smaller set"
        )

    if require_positive_label:
        positive = {r.query_id for r in qrels if r.grade > 0}
        missing = sorted(q.query_id for q in queries if q.query_id not in positive)
        if missing:
            raise EvalSetError(
                "these queries have no chunk graded above 0, so they measure nothing and "
                f"lower every mean uniformly: {missing}"
            )

    return EvalSet(tuple(queries), tuple(qrels), queries_path=qpath, qrels_path=rpath)


# --------------------------------------------------------------------------------------
# the harness
# --------------------------------------------------------------------------------------

#: What `evaluate` accepts back from a search function. A bare list of chunk ids, the
#: notebook's list of chunk dicts, a list of `ScoredChunk`, or a whole `RetrievalResult`.
SearchFn = Callable[..., Any]


def _ranked_ids(returned: Any) -> list[str]:
    """Normalise whatever a search function returned into a list of chunk ids, best first.

    Accepts a `RetrievalResult`, a sequence of `ScoredChunk`, a sequence of chunk dicts
    (the notebook's shape), or a sequence of bare ids. Raises rather than guessing.
    """
    if returned is None:
        return []
    results = getattr(returned, "results", None)
    if results is not None and not isinstance(returned, (str, bytes, dict)):
        returned = results
    if isinstance(returned, (str, bytes)):
        raise TypeError("search_fn returned a string; expected a sequence of results")

    ids: list[str] = []
    for item in returned:
        if isinstance(item, str):
            ids.append(item)
            continue
        chunk = getattr(item, "chunk", None)
        if chunk is not None:
            ids.append(chunk.chunk_id)
            continue
        if isinstance(item, Mapping):
            if "chunk_id" in item:
                ids.append(str(item["chunk_id"]))
                continue
            inner = item.get("chunk")
            if isinstance(inner, Mapping) and "chunk_id" in inner:
                ids.append(str(inner["chunk_id"]))
                continue
        cid = getattr(item, "chunk_id", None)
        if cid is not None:
            ids.append(str(cid))
            continue
        raise TypeError(f"cannot read a chunk_id out of a {type(item).__name__} result")
    return ids


def _call_search(search_fn: SearchFn, query: str, candidate_k: int) -> Any:
    """Call a search function, passing ``candidate_k`` only if it accepts it."""
    try:
        return search_fn(query, candidate_k=candidate_k)
    except TypeError as exc:
        if "candidate_k" not in str(exc):
            raise
        return search_fn(query)


def evaluate(
    search_fn: SearchFn,
    k: int = DEFAULT_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    label: str = "",
    *,
    eval_set: EvalSet | None = None,
    min_grade: int = DEFAULT_MIN_GRADE,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Score one retriever over the whole held-out set.

    Args:
        search_fn: Called as ``search_fn(query, candidate_k=candidate_k)`` (falling back
            to ``search_fn(query)`` if it takes no such keyword) and expected to return
            results best-first. See `_ranked_ids` for the accepted shapes.
        k: Top-k cutoff for nDCG and citation precision — what the user sees.
        candidate_k: Cutoff for recall — the size of the set handed to the reranker.
        label: Free-text run label, used in the printed line and as an MLflow run name.
        eval_set: A pre-loaded `EvalSet`. Defaults to `load_eval_set()`.
        min_grade: Hit threshold for MRR, recall and citation precision.
        verbose: Print the summary line, as the notebook did.

    Returns:
        ``(per_query, summary)``.

        ``per_query`` is a DataFrame with one row per query: ``query_id``, ``shape``,
        ``n_results``, ``n_labels``, and one column per metric named as in
        `metric_names`. Per-query rows are the deliverable, not a nicety — the mean tells
        you nothing actionable, and `failure_breakdown` needs the rows to separate a
        recall failure from a ranking failure.

        ``summary`` maps each metric key to its **NaN-skipping** mean, plus
        ``n_queries`` (rows evaluated) and ``n_<metric>`` (rows that actually counted
        toward that metric). Keys match `eval/thresholds_B.yaml`, so it can be handed
        straight to `compare`.
    """
    es = eval_set if eval_set is not None else load_eval_set()
    rel_by_query = es.rel_by_query
    names = metric_names(k=k, candidate_k=candidate_k)

    rows: list[dict[str, Any]] = []
    for q in es.queries:
        rel = rel_by_query.get(q.query_id, {})
        ranked = _ranked_ids(_call_search(search_fn, q.query, candidate_k))
        rows.append(
            {
                "query_id": q.query_id,
                "intent": q.intent,
                "n_results": len(ranked),
                "n_labels": len(rel),
                names["ndcg"]: ndcg_at_k(ranked, rel, k),
                names["mrr"]: mrr(ranked, rel, min_grade),
                names["recall"]: recall_at_k(ranked, rel, candidate_k, min_grade),
                names["citation"]: citation_precision(ranked, rel, k, min_grade),
            }
        )

    metric_cols = [names["ndcg"], names["mrr"], names["recall"], names["citation"]]
    columns = ["query_id", "intent", "n_results", "n_labels", *metric_cols]
    per_query = pd.DataFrame(rows, columns=columns)

    summary: dict[str, float] = {"n_queries": float(len(per_query))}
    for col in metric_cols:
        series = per_query[col] if col in per_query else pd.Series(dtype=float)
        summary[col] = float(series.mean(skipna=True)) if len(series) else NAN
        summary[f"n_{col}"] = float(series.notna().sum()) if len(series) else 0.0

    if verbose:
        shown = {c: round(summary[c], 3) for c in metric_cols}
        print(f"{label or 'run'}: {shown}  (n={int(summary['n_queries'])})")

    return per_query, summary


def failure_breakdown(
    per_query: pd.DataFrame,
    k: int = DEFAULT_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    ndcg_floor: float = 0.4,
) -> dict[str, Any]:
    """Split the failures into the two kinds that have opposite fixes.

    - ``recall_failures``: ``recall@candidate_k == 0``. The answer never entered the
      candidate set. Fix upstream, in chunking (02) or fusion (04); no amount of
      reranking recovers it.
    - ``ranking_failures``: recall above 0 but nDCG below ``ndcg_floor``. The answer was
      there and the ordering buried it. This is precisely what stage 08's DPO reranker
      is for.

    The ratio between the two counts is the argument for or against building the
    reranker at all, and it is the thing the mean nDCG cannot tell you.
    """
    names = metric_names(k=k, candidate_k=candidate_k)
    rec, nd = names["recall"], names["ndcg"]
    recall_fail = per_query[per_query[rec] == 0]
    rank_fail = per_query[(per_query[rec] > 0) & (per_query[nd] < ndcg_floor)]
    return {
        "recall_failures": recall_fail["query_id"].tolist(),
        "ranking_failures": rank_fail["query_id"].tolist(),
        "n_recall_failures": len(recall_fail),
        "n_ranking_failures": len(rank_fail),
        "unscored": per_query[per_query[rec].isna()]["query_id"].tolist(),
    }


# --------------------------------------------------------------------------------------
# the gate comparison
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricCheck:
    """The verdict on one metric. Carries the numbers, not just a boolean.

    A gate that says "failed" without saying which metric, by how much, and against
    which of the two rules is a gate nobody can act on.

    Attributes:
        metric: Key as it appears in `eval/thresholds_B.yaml` and in the summary.
        value: The candidate's value, or ``None`` when the candidate did not report it.
        floor: The absolute minimum from the threshold file, or ``None`` if unset.
        floor_shortfall: How far below the floor the candidate is; ``0.0`` when it
            passes, ``None`` when there is no floor to check.
        baseline: The champion's value, or ``None`` when not compared against a champion.
        regression_tolerance: Allowed drop vs the champion, or ``None`` if unset.
        delta: ``value - baseline``. Negative is a regression.
        regression_excess: How much further than the tolerance the value dropped; ``0.0``
            when within tolerance, ``None`` when no regression rule applied.
        passed: True when every rule that applied to this metric held.
        reasons: Human-readable failure reasons, empty when ``passed``.
    """

    metric: str
    value: float | None
    floor: float | None = None
    floor_shortfall: float | None = None
    baseline: float | None = None
    regression_tolerance: float | None = None
    delta: float | None = None
    regression_excess: float | None = None
    passed: bool = True
    reasons: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        return not self.passed

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, for a CI annotation or an MLflow tag."""
        return {
            "metric": self.metric,
            "value": self.value,
            "floor": self.floor,
            "floor_shortfall": self.floor_shortfall,
            "baseline": self.baseline,
            "regression_tolerance": self.regression_tolerance,
            "delta": self.delta,
            "regression_excess": self.regression_excess,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CompareVerdict:
    """The structured result of `compare`. This is what Sude's gate runner consumes.

    Truthy when the candidate may ship, so ``if not compare(...):`` works, but the
    fields are the point: `failures` names every metric that missed and by how much.

    Attributes:
        passed: True when no check failed.
        checks: One `MetricCheck` per metric named in the threshold file, in file order.
        missing_metrics: Metrics the threshold file requires that the candidate summary
            did not report (or reported as NaN). Whether these fail the run is decided
            by ``policy.fail_on_missing_metric``.
        compare_against: The effective ``policy.compare_against`` — ``"champion"`` when
            a baseline summary was supplied, ``"fixed_thresholds_only"`` otherwise.
    """

    passed: bool
    checks: tuple[MetricCheck, ...] = ()
    missing_metrics: tuple[str, ...] = ()
    compare_against: str = "fixed_thresholds_only"
    notes: tuple[str, ...] = field(default=())

    @property
    def failures(self) -> tuple[MetricCheck, ...]:
        """Every check that failed, with its numbers. Empty when `passed`."""
        return tuple(c for c in self.checks if c.failed)

    def __bool__(self) -> bool:
        return self.passed

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the whole verdict."""
        return {
            "passed": self.passed,
            "compare_against": self.compare_against,
            "checks": [c.as_dict() for c in self.checks],
            "failures": [c.metric for c in self.failures],
            "missing_metrics": list(self.missing_metrics),
            "notes": list(self.notes),
        }

    def report(self) -> str:
        """One-line-per-metric text report, suitable for a CI log."""
        lines = [f"gate: {'PASS' if self.passed else 'FAIL'} (vs {self.compare_against})"]
        for c in self.checks:
            mark = "ok  " if c.passed else "FAIL"
            val = "n/a" if c.value is None else f"{c.value:.4f}"
            detail = "; ".join(c.reasons) if c.reasons else ""
            base = (
                ""
                if c.baseline is None
                else f" (champion {c.baseline:.4f}, delta {c.delta:+.4f})"
            )
            lines.append(f"  [{mark}] {c.metric}={val}{base} {detail}".rstrip())
        return "\n".join(lines)


def _as_float(summary: Mapping[str, Any] | None, key: str) -> float | None:
    if not summary or key not in summary:
        return None
    try:
        v = float(summary[key])
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def compare(
    baseline_summary: Mapping[str, Any] | None,
    candidate_summary: Mapping[str, Any],
    thresholds: Mapping[str, Any] | None = None,
    *,
    section: str = "retrieval",
) -> CompareVerdict:
    """Decide whether a candidate retriever may be promoted over the champion.

    Implements the two rules `eval/thresholds_B.yaml` describes, in this order:

    1. **Absolute floor** (``min``). The candidate must clear it on its own merits.
       Notebook 06's rule is that each floor sits slightly below the measured baseline:
       a floor above your own baseline blocks every pull request including your own, and
       a floor far below it never fires.
    2. **Regression tolerance** (``regression_tolerance``) against the champion. A
       candidate may clear the floor and still be a step backwards; without this rule a
       run of small "still above the floor" regressions walks the system down to the
       floor and stops there. Metrics with no tolerance key — recall@20 — are floor-only
       by design: a ranking change must not move the ceiling at all.

    Args:
        baseline_summary: The champion's summary, or ``None``. ``None`` degrades the run
            to ``fixed_thresholds_only`` (correct for the very first run, when there is
            no champion yet) and records that in the verdict.
        candidate_summary: The challenger's summary, as returned by `evaluate`.
        thresholds: The parsed `eval/thresholds_B.yaml`. Defaults to `load_thresholds()`.
            Either the whole file or just its ``retrieval:`` block is accepted.
        section: Which block of the threshold file to enforce. ``"retrieval"`` here;
            Sude's runner passes ``"generation"`` for her own metrics.

    Returns:
        A `CompareVerdict`. Truthy on pass; ``verdict.failures`` names every metric that
        missed, with the floor shortfall and the regression delta attached.

    Note:
        ``policy.compare_against: fixed_thresholds_only`` in the config disables rule 2
        even when a baseline is supplied. ``policy.fail_on_missing_metric: true`` (the
        default) fails a run whose candidate summary is silently short a metric, which is
        the failure mode where a renamed metric key makes a gate quietly stop checking.
    """
    cfg = dict(thresholds) if thresholds is not None else load_thresholds()
    rules = cfg.get(section, cfg)
    if not isinstance(rules, Mapping):
        raise TypeError(f"thresholds[{section!r}] is not a mapping")
    policy = cfg.get("policy", {}) if isinstance(cfg.get("policy"), Mapping) else {}
    fail_on_missing = bool(policy.get("fail_on_missing_metric", True))

    want_champion = str(policy.get("compare_against", "champion")) == "champion"
    use_champion = want_champion and baseline_summary is not None
    effective = "champion" if use_champion else "fixed_thresholds_only"

    notes: list[str] = []
    if want_champion and baseline_summary is None:
        notes.append(
            "policy.compare_against is 'champion' but no baseline summary was supplied; "
            "regression tolerances were not enforced on this run"
        )

    checks: list[MetricCheck] = []
    missing: list[str] = []

    for metric, spec in rules.items():
        if not isinstance(spec, Mapping):
            continue
        value = _as_float(candidate_summary, metric)
        floor = spec.get("min")
        floor = float(floor) if floor is not None else None
        tol = spec.get("regression_tolerance")
        tol = float(tol) if tol is not None else None
        baseline = _as_float(baseline_summary, metric) if use_champion else None

        reasons: list[str] = []
        floor_shortfall: float | None = None
        regression_excess: float | None = None

        if value is None:
            missing.append(metric)
            reasons.append("candidate summary did not report this metric (missing or NaN)")
            checks.append(
                MetricCheck(
                    metric=metric,
                    value=None,
                    floor=floor,
                    baseline=baseline,
                    regression_tolerance=tol,
                    passed=not fail_on_missing,
                    reasons=tuple(reasons) if fail_on_missing else (),
                )
            )
            continue

        if floor is not None:
            floor_shortfall = max(0.0, floor - value)
            if value < floor:
                reasons.append(f"below floor {floor:.4f} by {floor_shortfall:.4f}")

        delta: float | None = None
        if baseline is not None:
            delta = value - baseline
            if tol is None:
                if delta < 0:
                    notes.append(
                        f"{metric} dropped {abs(delta):.4f} vs the champion; this metric has "
                        f"no regression_tolerance, so only its floor was enforced"
                    )
            else:
                regression_excess = max(0.0, -delta - tol)
                if regression_excess > 0:
                    reasons.append(
                        f"regressed {abs(delta):.4f} vs champion {baseline:.4f}, "
                        f"tolerance {tol:.4f}, over by {regression_excess:.4f}"
                    )

        checks.append(
            MetricCheck(
                metric=metric,
                value=value,
                floor=floor,
                floor_shortfall=floor_shortfall,
                baseline=baseline,
                regression_tolerance=tol,
                delta=delta,
                regression_excess=regression_excess,
                passed=not reasons,
                reasons=tuple(reasons),
            )
        )

    return CompareVerdict(
        passed=all(c.passed for c in checks),
        checks=tuple(checks),
        missing_metrics=tuple(missing),
        compare_against=effective,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------------------


def log_to_mlflow(
    summary: Mapping[str, Any],
    params: Mapping[str, Any] | None = None,
    *,
    run_name: str = "retrieval_eval_B",
    experiment: str = "retrieval_baseline_B",
    tracking_uri: str | None = None,
    per_query: pd.DataFrame | None = None,
    verdict: CompareVerdict | None = None,
    tags: Mapping[str, Any] | None = None,
) -> str | None:
    """Log one evaluation run to MLflow, so champion/challenger is a query and not a memory.

    Metrics alone are not enough to make a comparison honest: two runs with the same
    nDCG@5 mean different things if one used a different chunk size or fusion. The params
    are what make a stored run attributable, so log the ingestion and retrieval config
    alongside the numbers. The per-query frame is attached as an artifact because that is
    what the recall-vs-ranking failure split needs, and the mean cannot reconstruct it.

    Args:
        summary: The summary dict from `evaluate`. Non-finite values are skipped.
        params: Config values that produced these numbers (embed model, chunk strategy,
            fusion, candidate_k, top_k, ranker name).
        run_name: MLflow run name.
        experiment: MLflow experiment name.
        tracking_uri: Defaults to ``mlflow.tracking_uri`` in ``configs/reranker_B.yaml``,
            then to ``sqlite:///mlflow.db``.
        per_query: Logged as ``per_query.csv``.
        verdict: Logged as ``gate_verdict.json`` plus a ``gate_passed`` tag.
        tags: Extra MLflow tags.

    Returns:
        The MLflow run id, or ``None`` if mlflow is not installed (the harness is still
        usable without it; logging is not allowed to fail an evaluation).
    """
    try:
        import mlflow
    except ImportError:  # pragma: no cover - mlflow is a hard dependency in pyproject
        warnings.warn("mlflow is not installed; skipping run logging", RuntimeWarning, stacklevel=2)
        return None

    if tracking_uri is None:
        try:
            from research_assistant.config_J import load_config

            tracking_uri = load_config("reranker").get("mlflow", {}).get("tracking_uri")
        except Exception:
            tracking_uri = None
        tracking_uri = tracking_uri or "sqlite:///mlflow.db"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params({k: v for k, v in params.items() if v is not None})
        if tags:
            mlflow.set_tags(dict(tags))
        numeric = {}
        for key, value in summary.items():
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                numeric[key] = fv
        if numeric:
            mlflow.log_metrics(numeric)
        if per_query is not None:
            mlflow.log_text(per_query.to_csv(index=False), "per_query.csv")
        if verdict is not None:
            mlflow.set_tag("gate_passed", str(verdict.passed))
            mlflow.log_text(json.dumps(verdict.as_dict(), indent=2), "gate_verdict.json")
        return run.info.run_id


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

_EXAMPLE_QUERIES = "eval/datasets/queries_example_B.jsonl"
_EXAMPLE_QRELS = "eval/datasets/qrels_example_B.jsonl"


def _load_search_fn(dotted: str) -> SearchFn:
    """Import a ``module:attr`` search function, with an actionable error if it is missing."""
    mod_name, _, attr = dotted.partition(":")
    if not attr:
        mod_name, _, attr = dotted.rpartition(".")
    import importlib

    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        raise SystemExit(
            f"cannot import {mod_name!r} ({exc}). Stage 04 must be promoted to "
            f"src/research_assistant/retrieval/service_B.py before `make eval` can run "
            f"against the real retriever; use --stub to exercise the harness meanwhile."
        ) from exc
    fn = getattr(mod, attr, None)
    if fn is None:
        raise SystemExit(f"{mod_name!r} has no attribute {attr!r}")
    return fn


def _stub_search_fn(eval_set: EvalSet, *, drop_first: bool = False) -> SearchFn:
    """An oracle-ish stub retriever, for exercising the harness with no index built.

    Returns each query's labelled chunks in descending grade order, so the harness can be
    run end to end today. ``drop_first`` demotes the best chunk to last, which produces a
    plausible "ranking failure" run to compare against. This is a smoke-test device only
    — it reads the answer key and its numbers mean nothing about retrieval quality.
    """
    rel = eval_set.rel_by_query
    by_text = {q.query: q.query_id for q in eval_set.queries}

    def search(
        query: str, candidate_k: int = DEFAULT_CANDIDATE_K, **_: Any
    ) -> list[dict[str, str]]:
        qid = by_text.get(query)
        labels = rel.get(qid or "", {})
        ordered = [cid for cid, _g in sorted(labels.items(), key=lambda kv: -kv[1])]
        if drop_first and len(ordered) > 1:
            ordered = ordered[1:] + ordered[:1]
        return [{"chunk_id": cid} for cid in ordered[:candidate_k]]

    return search


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m eval.metrics.retrieval_B --config configs/retrieval_B.yaml`, per the Makefile."""
    ap = argparse.ArgumentParser(
        prog="python -m eval.metrics.retrieval_B",
        description="Score the retriever on the held-out set (Track A, stage 06).",
    )
    ap.add_argument("--config", default="configs/retrieval_B.yaml", help="retrieval config YAML")
    ap.add_argument("--queries", default=None, help="override the queries jsonl")
    ap.add_argument("--qrels", default=None, help="override the qrels jsonl")
    ap.add_argument(
        "--example",
        action="store_true",
        help="use eval/datasets/*_example_B.jsonl instead of the real held-out set",
    )
    ap.add_argument(
        "--stub",
        action="store_true",
        help="score a built-in stub retriever instead of importing the real one",
    )
    ap.add_argument("--stub-degraded", action="store_true", help="stub that buries the best chunk")
    ap.add_argument(
        "--search",
        default="research_assistant.retrieval.service_B:search",
        help="module:attr of the search function to score",
    )
    ap.add_argument("--k", type=int, default=None, help="top-k (default: config rerank.top_k)")
    ap.add_argument(
        "--candidate-k", type=int, default=None, help="default: config hybrid.candidate_k"
    )
    ap.add_argument("--label", default="baseline", help="run label")
    ap.add_argument("--min-queries", type=int, default=None, help="0 disables the size check")
    ap.add_argument("--mlflow", action="store_true", help="log this run to MLflow")
    ap.add_argument("--json", dest="as_json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args(argv)

    import yaml

    cfg_path = _resolve(args.config)
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open(encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    else:
        print(f"warning: {cfg_path} not found, falling back to defaults", file=sys.stderr)

    k = args.k or int(cfg.get("rerank", {}).get("top_k", DEFAULT_K))
    candidate_k = args.candidate_k or int(
        cfg.get("hybrid", {}).get("candidate_k", DEFAULT_CANDIDATE_K)
    )

    queries = args.queries or (_EXAMPLE_QUERIES if args.example else None)
    qrels = args.qrels or (_EXAMPLE_QRELS if args.example else None)
    min_queries = args.min_queries
    if min_queries is None and (args.example or queries):
        min_queries = 0

    eval_set = load_eval_set(queries, qrels, min_queries=min_queries)

    if args.stub or args.stub_degraded:
        search_fn: SearchFn = _stub_search_fn(eval_set, drop_first=args.stub_degraded)
    else:
        search_fn = _load_search_fn(args.search)

    per_query, summary = evaluate(
        search_fn, k=k, candidate_k=candidate_k, label=args.label, eval_set=eval_set, verbose=True
    )
    breakdown = failure_breakdown(per_query, k=k, candidate_k=candidate_k)

    if args.as_json:
        print(json.dumps({"summary": summary, "failures": breakdown}, indent=2))
    else:
        print()
        print(per_query.to_string(index=False))
        print()
        print(f"recall failures (fix in stages 02-04): {breakdown['n_recall_failures']}")
        print(f"ranking failures (fix in stage 08):    {breakdown['n_ranking_failures']}")

    verdict = compare(None, summary, section="retrieval")
    print()
    print(verdict.report())

    if args.mlflow:
        run_id = log_to_mlflow(
            summary,
            params={
                "fusion": cfg.get("hybrid", {}).get("fusion"),
                "candidate_k": candidate_k,
                "top_k": k,
                "ranker": cfg.get("rerank", {}).get("active"),
                "eval_queries": str(eval_set.queries_path),
            },
            run_name=args.label,
            per_query=per_query,
            verdict=verdict,
        )
        print(f"mlflow run: {run_id}")

    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
