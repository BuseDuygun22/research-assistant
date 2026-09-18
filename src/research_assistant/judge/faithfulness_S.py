"""Faithfulness and answer-relevance judges (Sude).

`passed` is computed here from counted violations against
`docs/editorial_rubric_S.md` — never read from the model's reply. The model
reports what it found; the rule decides what that means. The editor routes on the
result, so it has to be a rule anyone can audit rather than a model's opinion of
its own work.

Also here: `panel_faithfulness`, which runs several cheap judges instead of one
expensive one. The motivation is not cost. A single model shares its idiosyncratic
biases with itself by construction, so it can be highly self-consistent and
consistently wrong — reliability without validity, which is the failure a single
judge cannot detect about itself. Independent judges cannot share a bias that way,
and their disagreement is itself a usable uncertainty signal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median

from pydantic import BaseModel, ConfigDict, Field

from research_assistant.config_J import get_settings
from research_assistant.contracts.judge_J import (
    AnswerVerdict,
    FaithfulnessVerdict,
    FaithfulnessViolation,
    JudgeMeta,
)
from research_assistant.contracts.retrieval_J import RetrievedChunk
from research_assistant.llm_S import LLMClient, get_llm
from research_assistant.observability.tracing_S import KIND_JUDGE, span

PROMPT_PATH = Path(__file__).parent / "prompts" / "faithfulness_S.md"

# --- the rubric, as numbers --------------------------------------------------
# Mirrors docs/editorial_rubric_S.md §4. [D] — team decisions, revisable once the
# judge is calibrated and we know what they cost in escalation rate.
MIN_CITATION_PRECISION = 0.95
MIN_COVERAGE = 0.80


class _FaithfulnessReply(BaseModel):
    """What the model returns. Note the absence of `passed`: the model is not
    asked whether it passed, because it would be grading its own work."""

    model_config = ConfigDict(extra="forbid")

    violations: list[FaithfulnessViolation] = Field(default_factory=list)
    total_claims: int = Field(..., ge=0)
    cited_claims: int = Field(..., ge=0)
    supported_cited_claims: int = Field(..., ge=0)
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class _AnswerReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevance: int = Field(..., ge=0, le=3)
    evidence_sufficient: bool
    missing_aspects: list[str] = Field(default_factory=list)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    rationale: str = ""


def _meta(model: str | None = None) -> JudgeMeta:
    s = get_settings()
    return JudgeMeta(
        judge_model=model or s.judge_model,
        prompt_version=s.judge_prompt_version,
        temperature=0.0,
    )


def derive_passed(
    violations: Sequence[FaithfulnessViolation],
    citation_precision: float,
    coverage: float,
) -> bool:
    """The pass rule from `docs/editorial_rubric_S.md` §4, in code.

    A single deep violation fails the draft outright, regardless of how good the
    ratios look. That asymmetry is deliberate: a cited report containing one
    fabricated claim is worse than no report, because the citations lend the
    fabrication unearned credibility.
    """
    if any(v.is_deep for v in violations):
        return False
    if any(v.severity == "major" for v in violations):
        return False
    return citation_precision >= MIN_CITATION_PRECISION and coverage >= MIN_COVERAGE


def _ratios(reply: _FaithfulnessReply) -> tuple[float, float]:
    """Citation precision and coverage, with the degenerate cases pinned.

    `0/0` returns 0.0 rather than 1.0 in both. The vacuous-truth reading would
    score a draft that cites nothing as perfect, which makes "say nothing" the
    optimal strategy against the gate.
    """
    precision = (
        reply.supported_cited_claims / reply.cited_claims if reply.cited_claims else 0.0
    )
    coverage = reply.cited_claims / reply.total_claims if reply.total_claims else 0.0
    return min(1.0, precision), min(1.0, coverage)


def check_faithfulness(
    draft_id: str,
    draft: str,
    evidence: Sequence[RetrievedChunk],
    *,
    llm: LLMClient | None = None,
    model_name: str | None = None,
) -> FaithfulnessVerdict:
    """Grade one draft against the evidence it was written from."""
    context = "\n\n".join(
        f"[{r.chunk.chunk_id}] ({r.chunk.metadata.title})\n{r.chunk.text}" for r in evidence
    )
    with span("judge.faithfulness", KIND_JUDGE, draft_id=draft_id) as sp:
        reply = (llm or get_llm()).complete_json(
            system=PROMPT_PATH.read_text(encoding="utf-8"),
            user=f"RETRIEVED CONTEXT:\n{context}\n\nDRAFT:\n{draft}",
            schema=_FaithfulnessReply,
        )
        precision, coverage = _ratios(reply)
        passed = derive_passed(reply.violations, precision, coverage)
        sp.set(passed=passed, n_violations=len(reply.violations))
        return FaithfulnessVerdict(
            draft_id=draft_id,
            violations=list(reply.violations),
            citation_precision=precision,
            coverage=coverage,
            passed=passed,
            confidence=reply.confidence,
            meta=_meta(model_name),
        )


def check_answer(
    draft_id: str,
    query: str,
    draft: str,
    evidence: Sequence[RetrievedChunk],
    *,
    llm: LLMClient | None = None,
) -> AnswerVerdict:
    """Does the draft answer the question, and could the evidence have?

    Separate call from faithfulness rather than one combined prompt: bundling
    them invites the model to let a well-grounded draft slide on relevance, which
    is exactly the correlation the two verdicts exist to break.
    """
    context = "\n\n".join(f"[{r.chunk.chunk_id}] {r.chunk.text}" for r in evidence)
    system = (
        "You judge whether a draft answers the question asked, and whether the "
        "retrieved evidence could answer it at all.\n\n"
        "`relevance`: 0 does not address the question, 1 tangential, 2 partial, "
        "3 fully answers.\n"
        "`evidence_sufficient`: false when the retrieved passages cannot answer "
        "the question however well written the draft is. This is the abstention "
        "signal — set it false rather than rewarding a fluent summary of material "
        "that does not address the question.\n"
        "`missing_aspects`: parts of the question left unanswered, phrased as "
        "things to search for.\n"
        "Do not judge grounding or citation quality; that is graded separately.\n"
        "Return a single JSON object matching the schema."
    )
    with span("judge.answer", KIND_JUDGE, draft_id=draft_id) as sp:
        reply = (llm or get_llm()).complete_json(
            system=system,
            user=f"QUESTION:\n{query}\n\nRETRIEVED CONTEXT:\n{context}\n\nDRAFT:\n{draft}",
            schema=_AnswerReply,
        )
        sp.set(relevance=reply.relevance, sufficient=reply.evidence_sufficient)
        return AnswerVerdict(
            draft_id=draft_id,
            query=query,
            relevance=reply.relevance,  # type: ignore[arg-type]
            evidence_sufficient=reply.evidence_sufficient,
            missing_aspects=reply.missing_aspects,
            confidence=reply.confidence,
            rationale=reply.rationale,
            meta=_meta(),
        )


# --- the panel ---------------------------------------------------------------


@dataclass(frozen=True)
class PanelMember:
    """One judge in a panel: a client and the name to stamp on its verdict."""

    name: str
    client: LLMClient


def panel_faithfulness(
    draft_id: str,
    draft: str,
    evidence: Sequence[RetrievedChunk],
    members: Sequence[PanelMember],
) -> FaithfulnessVerdict:
    """Run several judges and aggregate. Returns one verdict, as if from one judge.

    Aggregation rules, each chosen for how it fails:

    * **`passed` by majority.** Not unanimity — one stubborn member should not be
      able to veto, and not any-fail, which would make the panel strictly more
      timid than its most timid member.
    * **Ratios by median, not mean.** One member misparsing the draft and
      reporting 0.0 precision should not drag the panel; the median ignores it.
    * **Violations by union, deduplicated.** Recall matters more than precision
      here, because a missed deep violation ships and a spurious one only costs a
      revision round.
    * **Confidence scaled by agreement.** This is the part worth having. When the
      panel splits, the aggregate confidence drops, which feeds the escalation
      rule — so disagreement between judges becomes a reason to involve a human
      rather than a number that gets averaged away.
    """
    if not members:
        raise ValueError("a panel needs at least one member")

    verdicts = [
        check_faithfulness(
            draft_id, draft, evidence, llm=m.client, model_name=m.name
        )
        for m in members
    ]
    n = len(verdicts)
    votes_passed = sum(1 for v in verdicts if v.passed)
    passed = votes_passed * 2 > n

    # Agreement in [0, 1]: 1.0 when unanimous, 0.0 on a perfect split.
    agreement = abs(votes_passed - (n - votes_passed)) / n
    confidence = fmean(v.confidence for v in verdicts) * agreement

    seen: set[tuple[str, str]] = set()
    merged: list[FaithfulnessViolation] = []
    for v in verdicts:
        for viol in v.violations:
            key = (viol.kind, viol.claim)
            if key not in seen:
                seen.add(key)
                merged.append(viol)

    return FaithfulnessVerdict(
        draft_id=draft_id,
        violations=merged,
        citation_precision=median(v.citation_precision for v in verdicts),
        coverage=median(v.coverage for v in verdicts),
        passed=passed,
        confidence=confidence,
        meta=_meta(model="panel:" + ",".join(m.name for m in members)),
    )
