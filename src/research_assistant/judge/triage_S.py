"""Pre-generation evidence triage (Sude).

Answers one question before the writer runs: *is this retrieved set worth writing
from?* Two things make it worth a separate call rather than a check on the draft.

**The failure it catches is confident, not hesitant.** Handed irrelevant passages,
a model does not hedge or stall — it writes a fluent, well-cited answer over them.
Nothing downstream looks wrong, because everything downstream only asks whether
the draft matches its sources, and it does. The passages were the problem.

**It is cheaper.** Judging the draft costs a writer call plus a faithfulness call
plus an answer call before the system learns what the passages alone would have
told it.

It also does the conflict detection, because the two questions share a reading of
the same passages and splitting them would double the cost for no benefit.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from research_assistant.config_J import get_settings
from research_assistant.contracts.judge_J import (
    EvidenceAssessment,
    EvidenceConflict,
    JudgeMeta,
)
from research_assistant.contracts.retrieval_J import RetrievedChunk
from research_assistant.llm_S import LLMClient, LLMError, get_llm
from research_assistant.observability.tracing_S import KIND_JUDGE, span
from research_assistant.prompts_S import with_examples

TRIAGE_SYSTEM = """You assess whether a set of retrieved passages can answer a question.

You are NOT writing an answer and NOT judging one. You are reading the passages
and reporting what they can and cannot support.

## label

- `sufficient` — a careful reader could answer the question from these passages.
- `partial` — they answer some of the question. Say what is missing.
- `insufficient` — they cannot answer it, however well written a draft might be.
  Choose this when the passages are on-topic but do not address what was asked;
  topical overlap is not an answer.
- `conflicting` — the passages contain claims that cannot all be true.

`conflicting` takes precedence over `sufficient`: if two passages disagree, say
so even when either one alone would answer the question.

## conflicts

Report every pair of passages that cannot both be right.

- `freshness` — one supersedes the other (a later study, a revised guideline, a
  corrected result). Set `newer_chunk_id`.
- `disagreement` — both are current and they genuinely disagree. This is normal
  in a research corpus and must be reported, not resolved by you.
- `incompatible_scope` — they only look contradictory because they measure
  different things, on different data, under different conditions.

Do not invent conflicts. Two passages emphasising different aspects of the same
finding are not in conflict.

## confidence

The probability a careful human using this rubric would assign the same label.
Use the full range; below 0.6 when the question is ambiguous or the passages are
borderline.

Return a single JSON object matching the schema."""


class _TriageReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    rationale: str = ""


_VALID_LABELS = {"sufficient", "partial", "insufficient", "conflicting"}


def assess_evidence(
    query: str,
    evidence: Sequence[RetrievedChunk],
    *,
    llm: LLMClient | None = None,
) -> EvidenceAssessment:
    """Triage a retrieved set. Never raises — a triage failure must not stop the run.

    An empty retrieval is `insufficient` without a model call: there is nothing to
    read, and spending a judge call to be told so is waste.

    If the judge itself fails, the assessment degrades to `partial` with zero
    confidence rather than blocking. Triage is an optimisation — it saves a wasted
    writer call — and an optimisation that can halt the pipeline is a liability.
    The downstream faithfulness and answer checks still run either way.
    """
    settings = get_settings()
    meta = JudgeMeta(
        judge_model=settings.judge_model,
        prompt_version=settings.judge_prompt_version,
        temperature=0.0,
    )

    if not evidence:
        return EvidenceAssessment(
            query=query,
            label="insufficient",
            confidence=1.0,
            rationale="retrieval returned nothing; no passages to assess",
            meta=meta,
        )

    context = "\n\n".join(
        f"[{r.chunk.chunk_id}] ({r.chunk.metadata.title}"
        f"{f', {r.chunk.metadata.year}' if r.chunk.metadata.year else ''})\n{r.chunk.text}"
        for r in evidence
    )
    with span("judge.triage", KIND_JUDGE, n_chunks=len(evidence)) as sp:
        try:
            reply = (llm or get_llm()).complete_json(
                system=with_examples(TRIAGE_SYSTEM, "triage"),
                user=f"QUESTION:\n{query}\n\nPASSAGES:\n{context}",
                schema=_TriageReply,
            )
        except (LLMError, ValueError):
            return EvidenceAssessment(
                query=query,
                label="partial",
                confidence=0.0,
                rationale="triage judge failed; proceeding without it",
                meta=meta,
            )

        label = reply.label if reply.label in _VALID_LABELS else "partial"

        # A judge that reports conflicts but labels the set `sufficient` has
        # contradicted itself. Trust the specific finding over the summary label:
        # it named the passages, which is harder to do by accident than picking a
        # label.
        conflicts = _drop_unknown(reply.conflicts, {r.chunk.chunk_id for r in evidence})
        if conflicts and label != "insufficient":
            label = "conflicting"

        sp.set(label=label, n_conflicts=len(conflicts))
        return EvidenceAssessment(
            query=query,
            label=label,  # type: ignore[arg-type]
            conflicts=conflicts,
            missing_aspects=reply.missing_aspects,
            confidence=reply.confidence,
            rationale=reply.rationale,
            meta=meta,
        )


def _drop_unknown(
    conflicts: Sequence[EvidenceConflict], known: set[str]
) -> list[EvidenceConflict]:
    """Discard conflicts naming passages that were not retrieved.

    A conflict between a real passage and a hallucinated id is not a conflict, and
    letting it through would route the run on evidence that does not exist.
    """
    return [c for c in conflicts if all(cid in known for cid in c.chunk_ids)]


def describe_conflicts(assessment: EvidenceAssessment) -> str:
    """Render conflicts for the writer's prompt.

    The writer needs the instruction as well as the finding: a disagreement must
    be *reported* rather than resolved, and a superseded result must be labelled
    rather than silently dropped. Left to itself a model picks one side, and the
    resulting draft is perfectly faithful to the passage it picked.
    """
    if not assessment.conflicts:
        return ""
    lines = ["The retrieved passages conflict. Report the conflict; do not resolve it."]
    for c in assessment.conflicts:
        if c.kind == "freshness":
            newer = c.newer_chunk_id or "the later one"
            lines.append(
                f"- SUPERSEDED: {c.summary} Lead with {newer} and say the earlier "
                f"result was superseded, rather than dropping it silently."
            )
        elif c.kind == "disagreement":
            lines.append(
                f"- DISAGREEMENT: {c.summary} Present both positions with their "
                f"citations. Do not pick a side."
            )
        else:
            lines.append(
                f"- SCOPE: {c.summary} These are not really in conflict — say what "
                f"each one measured."
            )
    return "\n".join(lines)
