"""Relevance judge (Sude).

Grades a retrieved chunk against a query on the TREC 0-3 scale, which is the same
scale Buse's qrels use — so judge verdicts and human labels are directly
comparable, which is what makes kappa calibration possible at all.

Two mitigations for known judge failure modes are structural rather than
prompt-level:

**Absolute grading, not pairwise.** Position bias in pairwise LLM judging is
measured at roughly 40% inconsistency for GPT-4-class models. Grading each chunk
independently on a fixed scale removes the comparison that bias attaches to.

**Deterministic shuffling where order exists.** When several chunks are graded in
one batch, the order is permuted by a seed derived from the query, and the seed is
logged. Order still exists — something has to go first — but it is reproducible
and uncorrelated with retrieval rank, so a judge that favours early positions
cannot systematically favour the retriever's own ordering.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from research_assistant.config_J import get_settings
from research_assistant.contracts.judge_J import JudgeMeta, PreferencePair, RelevanceVerdict
from research_assistant.contracts.retrieval_J import Chunk
from research_assistant.llm_S import get_llm
from research_assistant.observability.tracing_S import KIND_JUDGE, span
from research_assistant.prompts_S import with_examples

PROMPT_PATH = Path(__file__).parent / "prompts" / "relevance_S.md"


class _RelevanceReply(BaseModel):
    """What the model returns. Narrower than `RelevanceVerdict`, which also
    carries identity and provenance the model has no business inventing."""

    model_config = ConfigDict(extra="forbid")

    grade: int = Field(..., ge=0, le=3)
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str


def _prompt() -> str:
    return with_examples(PROMPT_PATH.read_text(encoding="utf-8"), "relevance")


def order_seed(query: str) -> int:
    """Deterministic per-query shuffle seed, logged with the verdict.

    Derived from the query rather than random so a re-run of the same eval set
    presents chunks in the same order. A judge whose order varies run to run adds
    variance to a promote/reject decision for no benefit.
    """
    return int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:8], 16)


def grade_chunk(query: str, chunk: Chunk) -> RelevanceVerdict:
    """Grade one chunk. No comparison set, by design."""
    settings = get_settings()
    with span("judge.relevance", KIND_JUDGE, chunk_id=chunk.chunk_id) as sp:
        reply = get_llm().complete_json(
            system=_prompt(),
            user=f"QUESTION:\n{query}\n\nPASSAGE ({chunk.chunk_id}):\n{chunk.text}",
            schema=_RelevanceReply,
        )
        sp.set(grade=reply.grade, confidence=reply.confidence)
        return RelevanceVerdict(
            query=query,
            chunk_id=chunk.chunk_id,
            grade=reply.grade,  # type: ignore[arg-type]
            confidence=reply.confidence,
            rationale=reply.rationale,
            meta=JudgeMeta(
                judge_model=settings.judge_model,
                prompt_version=settings.judge_prompt_version,
                temperature=0.0,
            ),
        )


def grade_chunks(query: str, chunks: Sequence[Chunk]) -> list[RelevanceVerdict]:
    """Grade a batch, presenting them in a deterministically shuffled order.

    Returns verdicts in the *input* order regardless, so callers do not have to
    care that the shuffle happened.
    """
    indexed = list(enumerate(chunks))
    random.Random(order_seed(query)).shuffle(indexed)
    graded: dict[int, RelevanceVerdict] = {}
    for original_index, chunk in indexed:
        graded[original_index] = grade_chunk(query, chunk)
    return [graded[i] for i in range(len(chunks))]


def build_preference_pairs(
    query: str,
    verdicts: Sequence[RelevanceVerdict],
    chunks: Sequence[Chunk],
    *,
    min_margin: float = 1.0,
) -> list[PreferencePair]:
    """Turn graded chunks into DPO pairs — the handoff to Buse's trainer.

    `min_margin` filters near-ties at construction, but the `margin` is retained
    on every pair so Buse can raise the threshold without regenerating the
    dataset. A 2-vs-3 pair usually reflects judge variance rather than a real
    preference, and filtering is cheaper than training through the noise.
    """
    by_id = {c.chunk_id: c for c in chunks}
    settings = get_settings()
    meta = JudgeMeta(
        judge_model=settings.judge_model,
        prompt_version=settings.judge_prompt_version,
        temperature=0.0,
    )
    pairs: list[PreferencePair] = []
    for chosen in verdicts:
        for rejected in verdicts:
            margin = float(chosen.grade - rejected.grade)
            if margin < min_margin:
                continue
            if chosen.chunk_id not in by_id or rejected.chunk_id not in by_id:
                continue
            pairs.append(
                PreferencePair(
                    query=query,
                    chosen_chunk_id=chosen.chunk_id,
                    chosen_text=by_id[chosen.chunk_id].text,
                    rejected_chunk_id=rejected.chunk_id,
                    rejected_text=by_id[rejected.chunk_id].text,
                    margin=margin,
                    source="judge",
                    meta=meta,
                )
            )
    return pairs
