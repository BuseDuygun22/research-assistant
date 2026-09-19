"""Fusion: combine the sparse and dense rankings into one candidate set.

Decides: reciprocal rank fusion is the default, weighted score sum is the documented
alternative behind the `hybrid.fusion` config key, and every fused candidate keeps
the rank it held in each arm.

RRF is the pick because the two arms produce scores on incomparable scales -- BM25 is
an unbounded term-weight sum, cosine distance is bounded in [0, 2] -- and rank-based
fusion needs no calibration between them. Weighted sum needs per-arm normalisation,
and that normalisation becomes another thing to tune and defend. Reasoning:
`notebooks/04_hybrid_retrieval_B.ipynb` (design-choice table: fusion).

Config: `configs/retrieval_B.yaml` -> `hybrid:`.

Deviation from the notebook prototype: `rrf()` there returns a bare list of ids, which
throws away which arm found what. `ScoredChunk.bm25_rank` / `.vector_rank` exist so a
trace can say which arm produced a hit, and an ablation needs exactly that signal, so
fusion here returns `FusedCandidate` records carrying both per-arm ranks. The notebook
also fuses only ids and so cannot implement weighted_sum at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..config_J import load_config

Fusion = str  # "rrf" | "weighted_sum"; validated against the contract's Literal


@dataclass(frozen=True)
class FusedCandidate:
    """One chunk id surviving fusion, with provenance.

    `bm25_rank` / `vector_rank` are 1-based within their arm, or None when that arm
    did not return the chunk at all.
    """

    chunk_id: str
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None

    @property
    def arms(self) -> tuple[str, ...]:
        """Which arms found this chunk. Useful for the stage-04 ablation table."""
        found = []
        if self.bm25_rank is not None:
            found.append("bm25")
        if self.vector_rank is not None:
            found.append("vector")
        return tuple(found)


def _rank_map(ids: Sequence[str]) -> dict[str, int]:
    """1-based rank per id, first occurrence wins (dedupe within an arm)."""
    ranks: dict[str, int] = {}
    for position, chunk_id in enumerate(ids, start=1):
        ranks.setdefault(chunk_id, position)
    return ranks


def reciprocal_rank_fusion(
    bm25_ids: Sequence[str],
    vector_ids: Sequence[str],
    *,
    k: int = 60,
) -> list[FusedCandidate]:
    """RRF: score = sum over arms of 1 / (k + rank).

    `k` damps the head of each list, so a chunk both arms rank moderately well beats
    a chunk one arm ranks first and the other misses entirely -- which is the whole
    reason to run two arms. Ties break on chunk_id so the ordering is deterministic.
    """
    bm25_ranks = _rank_map(bm25_ids)
    vector_ranks = _rank_map(vector_ids)

    candidates: list[FusedCandidate] = []
    for chunk_id in _ordered_union(bm25_ids, vector_ids):
        b_rank = bm25_ranks.get(chunk_id)
        v_rank = vector_ranks.get(chunk_id)
        score = 0.0
        if b_rank is not None:
            score += 1.0 / (k + b_rank)
        if v_rank is not None:
            score += 1.0 / (k + v_rank)
        candidates.append(
            FusedCandidate(chunk_id=chunk_id, score=score, bm25_rank=b_rank, vector_rank=v_rank)
        )
    candidates.sort(key=lambda c: (-c.score, c.chunk_id))
    return candidates


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    """Normalise one arm's scores to [0, 1]. A flat arm maps to all-1.0, not all-0.0.

    All-0.0 would silently delete an arm whose hits are all equally good; all-1.0
    keeps its weight intact, which is the conservative reading.
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo <= 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def weighted_sum_fusion(
    bm25_scored: Sequence[tuple[str, float]],
    vector_scored: Sequence[tuple[str, float]],
    *,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
) -> list[FusedCandidate]:
    """Weighted sum of per-arm min-max normalised scores.

    The dense arm arrives as `(chunk_id, distance)` where smaller is better, so it is
    converted to a similarity before normalisation. An arm that did not return a chunk
    contributes 0 for it -- this is precisely the asymmetry RRF avoids, and the reason
    RRF remains the default.
    """
    bm25_raw = {cid: s for cid, s in bm25_scored}
    vector_raw = {cid: -float(d) for cid, d in vector_scored}  # distance -> similarity

    bm25_norm = _min_max(bm25_raw)
    vector_norm = _min_max(vector_raw)

    bm25_ranks = _rank_map([cid for cid, _ in bm25_scored])
    vector_ranks = _rank_map([cid for cid, _ in vector_scored])

    candidates: list[FusedCandidate] = []
    for chunk_id in _ordered_union(list(bm25_raw), list(vector_raw)):
        score = (
            bm25_weight * bm25_norm.get(chunk_id, 0.0)
            + vector_weight * vector_norm.get(chunk_id, 0.0)
        )
        candidates.append(
            FusedCandidate(
                chunk_id=chunk_id,
                score=score,
                bm25_rank=bm25_ranks.get(chunk_id),
                vector_rank=vector_ranks.get(chunk_id),
            )
        )
    candidates.sort(key=lambda c: (-c.score, c.chunk_id))
    return candidates


def _ordered_union(*id_lists: Sequence[str]) -> list[str]:
    """Deduplicated union preserving first-seen order. This is the `dedupe_by: chunk_id`
    rule from the config, applied once, here, rather than in each arm."""
    seen: dict[str, None] = {}
    for ids in id_lists:
        for chunk_id in ids:
            seen.setdefault(chunk_id, None)
    return list(seen)


def fuse(
    bm25_scored: Sequence[tuple[str, float]],
    vector_scored: Sequence[tuple[str, float]],
    *,
    fusion: Fusion | None = None,
    rrf_k: int | None = None,
    weights: dict[str, float] | None = None,
    candidate_k: int | None = None,
) -> list[FusedCandidate]:
    """Fuse the two arms according to config, truncated to `candidate_k`.

    Both arms arrive as `(chunk_id, score)` so the same call site serves either
    strategy; RRF ignores the score values and uses only their order.

    `candidate_k` is a hard recall ceiling: anything dropped here can never be
    recovered by the reranker. That is why `recall_at_20` is a gate threshold.
    """
    if fusion is None or rrf_k is None or weights is None or candidate_k is None:
        cfg = load_config("retrieval")["hybrid"]
        fusion = fusion if fusion is not None else cfg["fusion"]
        rrf_k = rrf_k if rrf_k is not None else cfg["rrf_k"]
        weights = weights if weights is not None else cfg.get("weights", {})
        candidate_k = candidate_k if candidate_k is not None else cfg["candidate_k"]

    if fusion == "rrf":
        fused = reciprocal_rank_fusion(
            [cid for cid, _ in bm25_scored],
            [cid for cid, _ in vector_scored],
            k=int(rrf_k),
        )
    elif fusion == "weighted_sum":
        fused = weighted_sum_fusion(
            bm25_scored,
            vector_scored,
            bm25_weight=float(weights.get("bm25", 0.4)),
            vector_weight=float(weights.get("vector", 0.6)),
        )
    else:
        raise ValueError(f"unknown hybrid.fusion {fusion!r}, expected 'rrf' or 'weighted_sum'")

    return fused[: int(candidate_k)] if candidate_k else fused
