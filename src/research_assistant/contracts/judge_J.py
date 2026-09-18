"""Judge and preference-pair contracts (JOINT — Sude produces, both read).

One judge, three uses: it scores retrieved chunks (feeding DPO pair
construction), it checks drafts against their sources (feeding the editor), and
it checks drafts against the *question* (catching the draft that is perfectly
grounded and still does not answer anything). All three carry a rationale,
because a bare score cannot be audited and cannot be debugged when the judge
itself is wrong.

This module owns the *taxonomy* — what a violation is, and how deep it runs. It
deliberately does not own the *routing* — which node repairs it — because that
depends on budget state the contract knows nothing about. Routing lives in
`agents/routing_S.py`.

DRAFT — requires Buse's sign-off before either side builds against it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field


class JudgeMeta(BaseModel):
    """Stamped on every verdict. Without it, a metric shift cannot be attributed
    to the system or to a judge-prompt edit."""

    model_config = ConfigDict(extra="forbid")

    judge_model: str
    prompt_version: str
    temperature: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RelevanceVerdict(BaseModel):
    """Graded, not binary — nDCG needs gradations, and DPO pairs need a margin.

    Grades follow the TREC convention Buse's qrels also use:
    0 irrelevant, 1 marginal, 2 relevant, 3 directly answers.
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    chunk_id: str
    grade: Literal[0, 1, 2, 3]
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    meta: JudgeMeta


# --- the violation taxonomy --------------------------------------------------

ViolationKind = Literal[
    "unsupported_claim",
    "inferential_leap",
    "misattributed_citation",
    "overstated_certainty",
    "contradicts_source",
    "missing_citation",
]

SHALLOW_VIOLATION_KINDS: frozenset[str] = frozenset(
    {"missing_citation", "overstated_certainty", "misattributed_citation"}
)
"""Violations a rewrite can repair, because the evidence needed is already in the
writer's context and only its expression is wrong."""

DEEP_VIOLATION_KINDS: frozenset[str] = frozenset(
    {"unsupported_claim", "inferential_leap", "contradicts_source"}
)
"""Violations a rewrite cannot repair, because the evidence to repair them is not
in the writer's context.

The split is the one load-bearing consequence of the self-correction literature:
revision reliably fixes *shallow* errors and reliably compounds *deep* ones, so a
single undifferentiated revision edge is wrong in one direction or the other on
every pass. A writer told to fix an `unsupported_claim` without new evidence has
exactly two moves available — delete the claim, or invent support — and only one
of those is a behaviour we want. Deep violations therefore go back to retrieval,
not back to the keyboard.

`inferential_leap` is separated from `unsupported_claim` on purpose: 2026
evaluations of deep-research agents attribute most residual citation error to
improper inferential linking (the source supports A, the draft asserts B) rather
than to wholly uncited assertions. Same depth, different failure, different
rubric language — and the common case, so it earns its own label.
"""

_UNCLASSIFIED_KINDS = (
    frozenset(get_args(ViolationKind)) - DEEP_VIOLATION_KINDS - SHALLOW_VIOLATION_KINDS
)
if _UNCLASSIFIED_KINDS:  # pragma: no cover - import-time guard
    raise RuntimeError(
        f"violation kinds not classified by depth: {sorted(_UNCLASSIFIED_KINDS)}. "
        "Every kind must be in exactly one of SHALLOW_VIOLATION_KINDS or "
        "DEEP_VIOLATION_KINDS — an unclassified kind would otherwise fall through "
        "to the default route, which is the silent-misroute failure this check exists "
        "to prevent. Classify it, do not delete this guard."
    )

_OVERLAPPING_KINDS = DEEP_VIOLATION_KINDS & SHALLOW_VIOLATION_KINDS
if _OVERLAPPING_KINDS:  # pragma: no cover - import-time guard
    raise RuntimeError(
        f"violation kinds classified as both deep and shallow: {sorted(_OVERLAPPING_KINDS)}"
    )


class FaithfulnessViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ViolationKind
    claim: str = Field(..., description="The offending sentence, verbatim from the draft.")
    cited_chunk_ids: list[str] = Field(default_factory=list)
    explanation: str
    severity: Literal["minor", "major"] = "major"

    @property
    def is_classified(self) -> bool:
        """False for a kind that exists but has not been assigned a depth.

        Unreachable while the import-time guard above holds, and checked anyway:
        the routing layer treats an unclassified violation as an escalation
        rather than assuming a default, so the two guards fail in the same safe
        direction rather than relying on each other.
        """
        return self.kind in DEEP_VIOLATION_KINDS or self.kind in SHALLOW_VIOLATION_KINDS

    @property
    def is_deep(self) -> bool:
        """True when repair needs new evidence rather than new wording."""
        return self.kind in DEEP_VIOLATION_KINDS


class FaithfulnessVerdict(BaseModel):
    """Is the draft supported by its sources?

    `passed` is derived by the rubric, not by the model's self-report, because
    the editor routes on it and a model asked "did you pass?" is grading its own
    work.

    Scope warning, and it is the important one: this verdict measures
    *attribution*, not *truth*. A draft every one of whose claims traces cleanly
    to a retrieved span scores 1.0 here even when the source itself is wrong, the
    evidence was cherry-picked, or the draft never answers the question asked.
    `AnswerVerdict` covers the last of those; the first two are limits of the
    approach and are named in `docs/design_review_J.md` rather than papered over.
    """

    model_config = ConfigDict(extra="forbid")

    draft_id: str
    violations: list[FaithfulnessViolation] = Field(default_factory=list)
    citation_precision: float = Field(
        ..., ge=0.0, le=1.0, description="Cited claims actually supported / cited claims."
    )
    coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Claims carrying any citation / total claims. Coverage of the "
        "*draft* by citations — not coverage of the *corpus* by the draft, which is "
        "a different property and is measured by context recall on Buse's side.",
    )
    passed: bool
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Calibrated probability that `passed` is correct — not the "
        "model's self-reported certainty. Defaults to 1.0 so a judge that does "
        "not yet calibrate degrades to the previous always-trust behaviour "
        "instead of silently escalating every draft.",
    )
    meta: JudgeMeta

    @property
    def deep_violations(self) -> list[FaithfulnessViolation]:
        return [v for v in self.violations if v.is_deep]

    @property
    def shallow_violations(self) -> list[FaithfulnessViolation]:
        return [v for v in self.violations if v.is_classified and not v.is_deep]

    @property
    def unclassified_violations(self) -> list[FaithfulnessViolation]:
        """Violations with no depth assignment. Must escalate, never default."""
        return [v for v in self.violations if not v.is_classified]


class AnswerVerdict(BaseModel):
    """Does the draft answer the *question*, and could it have?

    Separate from `FaithfulnessVerdict` because the two properties are
    orthogonal and a system can max one while failing the other. The degenerate
    case is concrete: five verbatim quotes stitched together score 1.0
    faithfulness and 1.0 citation precision while answering nothing. Gating on
    faithfulness alone therefore rewards timidity, which is a Goodhart problem
    sitting directly in the promote/reject path.

    `evidence_sufficient` is the abstention signal. When the retrieved context
    cannot answer the question, the correct output is "the evidence does not
    answer this" — not a faithful summary of the nearest available material,
    which is the most convincing kind of wrong answer this system can produce.
    """

    model_config = ConfigDict(extra="forbid")

    draft_id: str
    query: str
    relevance: Literal[0, 1, 2, 3] = Field(
        ...,
        description="0 does not address the question, 1 tangential, 2 partial, "
        "3 fully answers. Same TREC-style scale as RelevanceVerdict so the two "
        "are readable side by side.",
    )
    evidence_sufficient: bool = Field(
        ...,
        description="False when the retrieved context cannot answer the question "
        "at all, whatever the draft says. Drives abstention.",
    )
    missing_aspects: list[str] = Field(
        default_factory=list,
        description="Parts of the question the draft leaves unanswered. Fed back "
        "to the researcher as reformulation targets, so the next retrieval round "
        "is aimed rather than merely repeated.",
    )
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    rationale: str = ""
    meta: JudgeMeta

    @property
    def answers_question(self) -> bool:
        """Grade 2 or better. The threshold is deliberately not 3: demanding a
        complete answer to every question would escalate every genuinely partial
        one, and 'partial but useful' is a legitimate output for a synthesis
        query over a finite corpus."""
        return self.relevance >= 2

    @property
    def should_abstain(self) -> bool:
        """Evidence cannot answer the question, so no amount of rewriting or
        re-retrieval will help. Distinct from `answers_question` being False,
        which may simply mean this draft missed something the corpus contains."""
        return not self.evidence_sufficient


class EvidenceConflict(BaseModel):
    """Two retrieved passages that cannot both be right.

    RAG implicitly assumes the retrieved set is mutually consistent, and for a
    corpus of research papers that assumption is simply false — superseded
    results and live disagreements are the normal case, not the edge case. With
    no conflict detection the writer silently picks one side, produces a draft
    that is perfectly faithful to the passage it chose, and the reader never
    learns the field is divided.

    `kind` drives what the writer should do, and the two want opposite handling:
    a superseded result should be reported as superseded; a genuine disagreement
    should be reported *as* a disagreement, not resolved by the writer.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["freshness", "disagreement", "incompatible_scope"] = Field(
        ...,
        description="'freshness' one passage supersedes the other; 'disagreement' "
        "both are current and they disagree; 'incompatible_scope' they answer "
        "different questions and only look contradictory.",
    )
    chunk_ids: list[str] = Field(..., min_length=2, description="The conflicting passages.")
    summary: str = Field(..., description="What they disagree about, in one sentence.")
    newer_chunk_id: str | None = Field(
        None, description="For 'freshness': which passage supersedes the other."
    )


class EvidenceAssessment(BaseModel):
    """Is this retrieved set worth writing from? Judged *before* the writer runs.

    Two findings motivate assessing retrieval before generation rather than
    judging the draft afterwards:

    * A model handed irrelevant passages writes a fluent answer over them anyway
      — the failure is confident, not hesitant, so nothing downstream looks wrong.
    * Discovering it after the fact costs a full writer call plus a judge call
      before the system learns what it could have known from the passages alone.

    `label` is deliberately not a boolean. `partial` and `conflicting` are real
    states with different correct responses, and collapsing them into
    "insufficient" throws away the distinction between "retrieve more" and
    "report the disagreement".
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    label: Literal["sufficient", "partial", "insufficient", "conflicting"] = Field(
        ...,
        description="'sufficient' answers the question; 'partial' answers some of "
        "it; 'insufficient' cannot answer it; 'conflicting' contains evidence that "
        "cannot all be true.",
    )
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    missing_aspects: list[str] = Field(
        default_factory=list, description="Fed to the researcher as reformulation targets."
    )
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    rationale: str = ""
    meta: JudgeMeta

    @property
    def can_write(self) -> bool:
        """Enough to draft from. `conflicting` qualifies — the correct output is a
        draft that reports the disagreement, not silence."""
        return self.label in ("sufficient", "partial", "conflicting")

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)


class PreferencePair(BaseModel):
    """One DPO training example: (query, chosen, rejected).

    `margin` is retained so training can filter out near-ties. Pairs built from a
    thin grade gap are the main source of label noise in judge-generated DPO data,
    and dropping them is cheaper than trying to train through them.
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    chosen_chunk_id: str
    chosen_text: str
    rejected_chunk_id: str
    rejected_text: str
    margin: float = Field(..., ge=0.0, description="chosen.grade - rejected.grade.")
    source: Literal["judge", "human"] = "judge"
    meta: JudgeMeta
