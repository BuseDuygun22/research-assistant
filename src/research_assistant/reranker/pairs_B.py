"""Preference-pair construction — Track A / Buse, promoted from notebook 07.

The handoff
-----------
Sude's relevance judge grades (query, chunk) pairs 0-3 (TREC scale, same as
Buse's qrels) and emits `contracts.judge_J.RelevanceVerdict` rows. That schema
is this module's *input* contract. This module turns those rows into
`contracts.judge_J.PreferencePair` triples — the shared shape both this
trainer and Sude's leakage test (`tests/test_no_eval_leakage_J.py`) read —
which is what DPO consumes in stage 08.

`RelevanceVerdict` carries `chunk_id` but not the chunk's text, so pair
construction resolves text through `chunk_lookup` (defaults to the BM25
catalogue, which already holds every `Chunk` in memory) before a
`PreferencePair` can be built. A verdict whose chunk_id no longer resolves
(the index was rebuilt since the judge ran) is dropped rather than guessed.

Why the rules are the rules
---------------------------
* **Minimum score gap.** The judge is an LLM; its own test-retest jitter is
  worth something. A pair whose gap is at or below that jitter trains the
  reranker on the judge's noise, not on relevance. On the 0-3 scale the right
  number is picked empirically — see `scripts/sweep_min_score_gap_B.py` and
  the comment on `pairs.min_score_gap` in `configs/reranker_B.yaml` for the
  pair-yield analysis behind the current value — rather than carried over
  from the old 1-5 scale's tuning, where a gap of 2 out of 5 does not mean
  the same thing as a gap of 2 out of 3.
* **Pairs per query are capped (4).** Candidate sets are not uniform in size: one
  broad query with 20 judged candidates can contribute hundreds of combinations
  while a narrow one contributes two. Uncapped, the loss is dominated by a handful
  of verbose queries and the model learns their idiosyncrasies rather than
  ranking. The cap makes every query contribute comparable signal.
* **Rejected chunks must be hard negatives**, drawn from the same query's own
  candidate set. A "relevant chunk vs. a random chunk from another paper" pair is
  trivially separable — lexical overlap alone settles it — so the model can
  minimise the loss without learning anything the fusion ranker did not already
  know. The useful gradient is the one that separates *plausible* from *actually
  answering*, and only same-candidate-set negatives produce it.
* **Dedupe.** The judge may score the same chunk under near-identical queries;
  duplicated triples silently reweight those examples.
* **The leakage guard.** See `assert_no_eval_leakage` — it is the most important
  function in this file. If a held-out eval query reaches training, every number
  produced downstream (stage 06 metrics, the promotion gate, the report) is a lie,
  and it is a lie that looks like success. The guard therefore raises rather than
  warns, and runs before any pair is built.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from research_assistant.contracts.judge_J import PreferencePair, RelevanceVerdict
from research_assistant.reranker.registry_B import load_reranker_config, resolve_path

__all__ = [
    "PreferencePair",
    "EvalLeakageError",
    "ChunkLookup",
    "normalise_query",
    "assert_no_eval_leakage",
    "load_judge_scores",
    "load_eval_queries",
    "default_chunk_lookup",
    "build_pairs",
    "write_pairs",
    "build_pairs_from_config",
]

_WHITESPACE = re.compile(r"\s+")

#: Resolves a chunk_id to its text, or None if the chunk no longer exists.
ChunkLookup = Callable[[str], "str | None"]


class EvalLeakageError(RuntimeError):
    """A held-out evaluation query was found in the training data."""


def normalise_query(query: str) -> str:
    """Canonical form used for leakage comparison and grouping.

    Case and whitespace differences must not let a held-out query slip past the
    guard, so comparison is done on this form, never on the raw string.
    """
    return _WHITESPACE.sub(" ", query.strip().lower())


# --------------------------------------------------------------------------- #
# the leakage guard
# --------------------------------------------------------------------------- #
def assert_no_eval_leakage(
    training_queries: Iterable[str],
    eval_queries: Iterable[str],
) -> None:
    """Raise `EvalLeakageError` if any held-out eval query appears in training.

    Comparison is on `normalise_query`, so "What is RAG?" and "  what is rag? "
    are the same query. The failure mode this prevents is the one that does not
    look like a failure: a reranker trained on its own test set scores well,
    passes the gate, and ships — and the report's headline number means nothing.
    Warnings get ignored in a Friday evening run, so this raises.
    """
    eval_set = {normalise_query(q) for q in eval_queries if q and q.strip()}
    if not eval_set:
        return
    leaked = sorted({normalise_query(q) for q in training_queries if q and q.strip()} & eval_set)
    if leaked:
        preview = ", ".join(repr(q) for q in leaked[:5])
        more = f" (+{len(leaked) - 5} more)" if len(leaked) > 5 else ""
        raise EvalLeakageError(
            f"{len(leaked)} held-out eval query/queries leaked into the training data: "
            f"{preview}{more}. Drop these queries from the judge scores, or drop them "
            "from eval/datasets/queries_B.jsonl — but every metric computed before you "
            "do is invalid."
        )


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_judge_scores(path: str | Path, *, strict: bool = True) -> list[RelevanceVerdict]:
    """Read Sude's judge output (JSONL) and validate it against the joint contract.

    Validating here rather than at training time means a judge-schema drift fails
    in seconds, at the handoff, instead of after a training run.
    """
    rows: list[RelevanceVerdict] = []
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"judge scores not found at {file}. This file is produced by Sude's relevance "
            "judge; `pairs.judge_scores_path` in configs/reranker_B.yaml points at it."
        )
    for lineno, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(RelevanceVerdict.model_validate_json(line))
        except ValidationError as exc:
            if strict:
                raise ValueError(f"{file}:{lineno} does not match RelevanceVerdict: {exc}") from exc
    return rows


def load_eval_queries(path: str | Path) -> list[str]:
    """Read the held-out query strings from `eval/datasets/queries_B.jsonl`.

    Accepts `query` or `text` as the field name; a missing file is an error, not
    an empty blocklist, because an empty blocklist would make the guard a no-op.
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"eval query blocklist not found at {file}. Refusing to build pairs without it: "
            "an absent blocklist silently disables the leakage guard."
        )
    queries: list[str] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get("query") or row.get("text")
        if value:
            queries.append(str(value))
    return queries


def default_chunk_lookup() -> ChunkLookup:
    """Resolve chunk text from the BM25 catalogue, which already holds every
    indexed `Chunk` in memory. Lazy import: this module must stay importable
    without a built index (e.g. for the leakage-guard unit tests)."""
    from research_assistant.retrieval.bm25_B import BM25Index

    index = BM25Index.load()

    def _lookup(chunk_id: str) -> str | None:
        chunk = index.by_id.get(chunk_id)
        return chunk.text if chunk is not None else None

    return _lookup


def write_pairs(pairs: Sequence[PreferencePair], path: str | Path) -> Path:
    """Write the triples as JSONL, one pair per line."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(pair.model_dump_json() + "\n")
    return out


# --------------------------------------------------------------------------- #
# pair construction
# --------------------------------------------------------------------------- #
def build_pairs(
    scores: Sequence[RelevanceVerdict],
    *,
    chunk_lookup: ChunkLookup | None = None,
    min_score_gap: int = 1,
    max_pairs_per_query: int = 4,
    hard_negatives_only: bool = True,
    dedupe: bool = True,
) -> list[PreferencePair]:
    """Turn judged rows into `PreferencePair` triples.

    Rows are grouped by the *normalised* query so that trivial spelling variants
    share a candidate set. Within a group, candidates are sorted by descending
    grade and every high/low combination is considered in that order, so the
    per-query cap keeps the largest gaps — the clearest training signal — rather
    than an arbitrary slice.

    With `hard_negatives_only=False` the rejected side may come from another
    query's candidate set. That mode exists only as the ablation for the ledger;
    it is not the recommended setting (see the module docstring).

    `chunk_lookup` resolves `chunk_id` to text (`RelevanceVerdict` does not carry
    it); a verdict whose chunk no longer resolves is skipped, not guessed at.
    """
    if min_score_gap < 1:
        raise ValueError("min_score_gap must be >= 1; a gap of 0 is not a preference")
    if max_pairs_per_query < 1:
        raise ValueError("max_pairs_per_query must be >= 1")
    lookup = chunk_lookup if chunk_lookup is not None else default_chunk_lookup()

    grouped: dict[str, list[RelevanceVerdict]] = defaultdict(list)
    for row in scores:
        grouped[normalise_query(row.query)].append(row)

    pairs: list[PreferencePair] = []
    seen: set[tuple[str, str, str]] = set()
    text_cache: dict[str, str | None] = {}

    def text_of(chunk_id: str) -> str | None:
        if chunk_id not in text_cache:
            text_cache[chunk_id] = lookup(chunk_id)
        return text_cache[chunk_id]

    for key in sorted(grouped):
        rows = grouped[key]
        # Stable, deterministic ordering: grade desc, then chunk_id for ties.
        candidates = sorted(rows, key=lambda r: (-r.grade, r.chunk_id))
        display_query = candidates[0].query
        negatives: list[RelevanceVerdict] = candidates
        if not hard_negatives_only:
            # Ablation only: allow any lower-graded row from the whole corpus.
            negatives = sorted(scores, key=lambda r: (r.grade, r.chunk_id))

        made = 0
        for chosen, rejected in _candidate_combinations(candidates, negatives, hard_negatives_only):
            if made >= max_pairs_per_query:
                break
            gap = chosen.grade - rejected.grade
            if gap < min_score_gap:
                continue
            if chosen.chunk_id == rejected.chunk_id:
                continue
            dedupe_key = (key, chosen.chunk_id, rejected.chunk_id)
            if dedupe and dedupe_key in seen:
                continue
            chosen_text = text_of(chosen.chunk_id)
            rejected_text = text_of(rejected.chunk_id)
            if chosen_text is None or rejected_text is None:
                continue
            seen.add(dedupe_key)
            pairs.append(
                PreferencePair(
                    query=display_query,
                    chosen_chunk_id=chosen.chunk_id,
                    chosen_text=chosen_text,
                    rejected_chunk_id=rejected.chunk_id,
                    rejected_text=rejected_text,
                    margin=float(gap),
                    source="judge",
                    meta=chosen.meta,
                )
            )
            made += 1
    return pairs


def _candidate_combinations(
    candidates: Sequence[RelevanceVerdict],
    negatives: Sequence[RelevanceVerdict],
    hard_negatives_only: bool,
) -> Iterable[tuple[RelevanceVerdict, RelevanceVerdict]]:
    """Yield (chosen, rejected) candidates, largest score gap first."""
    if hard_negatives_only:
        # `candidates` is already grade-descending, so combinations() walks the
        # widest gaps first for each chosen row.
        yield from itertools.combinations(candidates, 2)
        return
    for chosen in candidates:
        for rejected in negatives:
            yield chosen, rejected


# --------------------------------------------------------------------------- #
# orchestration (what the CLI calls)
# --------------------------------------------------------------------------- #
def build_pairs_from_config(
    config: dict[str, Any] | None = None,
    *,
    config_path: str | Path | None = None,
    write: bool = True,
) -> list[PreferencePair]:
    """Load judge scores, run the leakage guard, build pairs, optionally write them."""
    cfg = config if config is not None else load_reranker_config(config_path)
    pcfg = cfg["pairs"]

    scores = load_judge_scores(resolve_path(pcfg["judge_scores_path"]))
    eval_queries = load_eval_queries(resolve_path(pcfg["eval_query_blocklist"]))

    # Guard first. Nothing is built, and nothing is written, if this raises.
    assert_no_eval_leakage((row.query for row in scores), eval_queries)

    pairs = build_pairs(
        scores,
        min_score_gap=int(pcfg.get("min_score_gap", 2)),
        max_pairs_per_query=int(pcfg.get("max_pairs_per_query", 4)),
        hard_negatives_only=bool(pcfg.get("hard_negatives_only", True)),
        dedupe=bool(pcfg.get("dedupe", True)),
    )
    if write:
        write_pairs(pairs, resolve_path(pcfg["out_path"]))
    return pairs
