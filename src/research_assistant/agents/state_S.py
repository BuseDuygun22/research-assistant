"""Shared agent state (Sude).

One object every node reads and a controlled set of fields each node may write.
The alternative — nodes passing ad-hoc dicts — fails in a specific and expensive
way: two nodes disagree about what "the draft" currently is, and the verdict that
routing depends on turns out to have been computed against a stale one.

Three rules encoded here:

**Write access is declared, not assumed.** `WRITABLE_BY` names which node may
change which field, and `apply` enforces it. A writer that silently overwrote
`evidence` would invalidate every verdict computed from it, and the resulting bug
would appear as an inexplicable routing decision several steps later.

**Provenance is per claim, not per draft.** `ClaimSpan` links one atomic claim to
the span supporting it, because that is the granularity the faithfulness judge
works at and the granularity a citation repair needs. Storing "the draft" and
"the sources" as two opaque blobs makes every verification a whole-document
comparison, which is both expensive and exactly where long-context accuracy is
worst.

**History is kept, not compacted.** Prior drafts and verdicts stay addressable by
id. Summarising them to save prompt space is the one context-engineering move the
2026 work is most consistently against — it is lossy, irreversible, and drops
constraints stated early. Nodes are handed the slice they need; nothing is
destroyed to make that slice small.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from research_assistant.contracts.judge_J import (
    AnswerVerdict,
    EvidenceAssessment,
    FaithfulnessVerdict,
)
from research_assistant.contracts.retrieval_J import RetrievedChunk

from .routing_S import Budget, RoutingDecision

NodeName = Literal["researcher", "writer", "editor", "flag_for_human"]

WRITABLE_BY: dict[str, frozenset[NodeName]] = {
    "evidence": frozenset({"researcher"}),
    "queries_issued": frozenset({"researcher"}),
    "draft": frozenset({"writer"}),
    "claims": frozenset({"writer"}),
    "assessment": frozenset({"editor"}),
    "faithfulness": frozenset({"editor"}),
    "answer": frozenset({"editor"}),
    "decisions": frozenset({"editor"}),
    "budget": frozenset({"editor"}),
    "outcome": frozenset({"editor", "flag_for_human"}),
}
"""Which node may write which field. Anything absent is read-only to every node
and is set once at construction (`question`, `run_id`)."""


Outcome = Literal["pending", "accepted", "abstained", "escalated"]

_TERMINAL_OUTCOME: dict[str, Outcome] = {
    "accept": "accepted",
    "abstain": "abstained",
    "escalate": "escalated",
}
"""Routes that end a run, and the outcome each records. Every other route leaves
the run pending."""


class StateViolation(RuntimeError):
    """A node tried to write a field it does not own."""


@dataclass(frozen=True)
class ClaimSpan:
    """One atomic claim and the span that supports it.

    Claim-level rather than sentence-level: a sentence can carry two claims of
    which only one is supported, and a verifier that can only accept or reject
    whole sentences cannot express that. Decomposition is what makes
    `citation_precision` computable rather than impressionistic.
    """

    claim_id: str
    text: str
    chunk_id: str | None = None
    quote: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    verified: bool | None = None

    @property
    def is_cited(self) -> bool:
        return self.chunk_id is not None


@dataclass(frozen=True)
class Draft:
    """One draft, kept rather than overwritten.

    Revision preserves verified claims instead of regenerating the whole answer:
    regenerating correct prose is both wasted tokens and a fresh opportunity to
    introduce a violation into a passage that was already fine.
    """

    draft_id: str
    text: str
    claims: tuple[ClaimSpan, ...] = ()
    revision_of: str | None = None

    @property
    def verified_claims(self) -> tuple[ClaimSpan, ...]:
        return tuple(c for c in self.claims if c.verified is True)

    @property
    def unverified_claims(self) -> tuple[ClaimSpan, ...]:
        return tuple(c for c in self.claims if c.verified is not True)


@dataclass(frozen=True)
class AgentState:
    """The whole run, in one object.

    Frozen: every mutation goes through `apply`, which returns a new state. That
    makes the trajectory a list of states rather than a mutable blob whose
    history is gone by the time something looks wrong.
    """

    run_id: str
    question: str
    evidence: tuple[RetrievedChunk, ...] = ()
    queries_issued: tuple[str, ...] = ()
    drafts: tuple[Draft, ...] = ()
    assessment: EvidenceAssessment | None = None
    faithfulness: FaithfulnessVerdict | None = None
    answer: AnswerVerdict | None = None
    decisions: tuple[RoutingDecision, ...] = ()
    budget: Budget = field(default_factory=Budget)
    outcome: Outcome = "pending"

    # --- reads ---------------------------------------------------------------

    @property
    def draft(self) -> Draft | None:
        """The current draft. `drafts` keeps the earlier ones."""
        return self.drafts[-1] if self.drafts else None

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(r.chunk.chunk_id for r in self.evidence)

    @property
    def is_terminal(self) -> bool:
        return self.outcome != "pending"

    def repair_brief(self) -> dict[str, Any]:
        """What the writer needs to fix *this* draft — and nothing else.

        Targeted correction rather than "regenerate the answer": the writer gets
        the specific claims that failed and the diagnosis for each, so verified
        prose survives the revision untouched. Handing back the whole draft with
        "try again" is what makes revision compound errors instead of repairing
        them.
        """
        if self.faithfulness is None or self.draft is None:
            return {"claims_to_fix": [], "keep": [], "guidance": []}
        broken = {v.claim: v for v in self.faithfulness.violations}
        return {
            "claims_to_fix": [
                {
                    "claim": c.text,
                    "claim_id": c.claim_id,
                    "kind": broken[c.text].kind,
                    "explanation": broken[c.text].explanation,
                    "cited_chunk_ids": broken[c.text].cited_chunk_ids,
                }
                for c in self.draft.claims
                if c.text in broken
            ],
            "keep": [c.claim_id for c in self.draft.claims if c.text not in broken],
            "guidance": (self.answer.missing_aspects if self.answer else []),
        }

    def retrieval_brief(self) -> dict[str, Any]:
        """What the researcher needs for another round — what is missing, and
        what has already been tried, so a reformulation explores rather than
        repeats."""
        deep = self.faithfulness.deep_violations if self.faithfulness else []
        return {
            "question": self.question,
            "unsupported_claims": [v.claim for v in deep],
            "missing_aspects": (
                (self.answer.missing_aspects if self.answer else [])
                + (self.assessment.missing_aspects if self.assessment else [])
            ),
            "already_tried": list(self.queries_issued),
            "already_retrieved": sorted(self.evidence_ids),
        }

    # --- writes --------------------------------------------------------------

    def apply(self, node: NodeName, **changes: Any) -> AgentState:
        """Return a new state with `changes` applied, if `node` may make them."""
        for name in changes:
            allowed = WRITABLE_BY.get(name)
            if allowed is None:
                raise StateViolation(
                    f"'{name}' is not a writable field; it is set at construction "
                    f"or does not exist. Writable: {sorted(WRITABLE_BY)}"
                )
            if node not in allowed:
                raise StateViolation(
                    f"node '{node}' may not write '{name}' (owned by "
                    f"{sorted(allowed)}). This guard exists because a field written "
                    "by the wrong node invalidates every verdict derived from it, "
                    "and the resulting bug surfaces several steps away from its cause."
                )
        # `claims` is addressed as a field for permission purposes but lives on
        # the current draft, so route it there explicitly.
        if "claims" in changes:
            claims = changes.pop("claims")
            if not self.drafts:
                raise StateViolation("cannot set claims before a draft exists")
            head = replace(self.drafts[-1], claims=tuple(claims))
            changes["drafts"] = (*self.drafts[:-1], head)
        return replace(self, **changes)

    def add_draft(self, draft: Draft) -> AgentState:
        """Append a draft, preserving every earlier one."""
        return replace(self, drafts=(*self.drafts, draft))

    def record(self, decision: RoutingDecision) -> AgentState:
        """Log a routing decision and charge its cost to the budget.

        Decision and budget move together on purpose: a route recorded without
        its cost charged is how a loop escapes its own budget.
        """
        return replace(
            self,
            decisions=(*self.decisions, decision),
            budget=self.budget.spend(decision.route),
            outcome=_TERMINAL_OUTCOME.get(decision.route, self.outcome),
        )

    # --- trajectory readout --------------------------------------------------

    def trajectory(self) -> list[dict[str, Any]]:
        """The run as a list of decisions, for tracing and for the eval gate.

        This is the raw material for measuring the *process* rather than only the
        final answer: how often routing sent work to each node, how many rounds
        converged, where escalations came from.
        """
        return [
            {
                "step": i,
                "route": d.route,
                "node": d.node,
                "trigger": d.trigger,
                "reason": d.reason,
                "rewrites_used": d.budget.rewrites_used,
                "re_retrievals_used": d.budget.re_retrievals_used,
            }
            for i, d in enumerate(self.decisions)
        ]
