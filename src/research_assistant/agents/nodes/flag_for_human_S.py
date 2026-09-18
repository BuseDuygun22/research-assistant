"""Human escalation node (Sude).

Terminal. Produces a handover a person can act on without re-running anything:
what was asked, what was retrieved, what the draft said, what failed, and — the
part that actually saves time — *why the machine stopped*, taken from the routing
decision rather than inferred.

Escalation is not failure. Eight distinct triggers reach here and they want
different human responses: an unclassified violation is a bug report, a low
confidence verdict is a labelling task, an exhausted budget is a tuning question,
and insufficient evidence is a corpus gap. Collapsing them into "the agent gave
up" throws away a diagnosis the system already computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research_assistant.observability.tracing_S import KIND_AGENT, span

from ..state_S import AgentState

NEXT_STEPS: dict[str, str] = {
    "unclassified_violation": (
        "Bug: the judge emitted a violation kind with no depth assignment. "
        "Classify it in contracts/judge_J.py before this can route automatically."
    ),
    "unexplained_failure": (
        "The rubric failed the draft without itemising why, or the judge did not "
        "return a usable verdict. Check the judge prompt and the rubric together."
    ),
    "low_faithfulness_confidence": (
        "The judge was unsure. Hand-label this one — cases near the rubric "
        "boundary are the most informative additions to the calibration set."
    ),
    "low_answer_confidence": (
        "The judge was unsure whether the draft answers the question. Hand-label "
        "and add to the calibration set."
    ),
    "evidence_insufficient": (
        "The corpus appears not to contain an answer. Either the question is "
        "genuinely unanswerable — label it intent='unanswerable' and it becomes a "
        "useful eval case — or the corpus has a gap worth filling."
    ),
    "rewrite_budget_exhausted": (
        "Shallow violations survived every rewrite. Usually a prompt problem "
        "rather than an evidence problem: read the repair briefs and see what the "
        "writer was actually told."
    ),
    "re_retrieval_budget_exhausted": (
        "Retrieval could not find supporting evidence in the allowed rounds. "
        "Check whether the reformulations were genuinely different from each other."
    ),
    "cost_ceiling": (
        "Cost-aware escalation fired: finishing the remaining budget would have "
        "cost more than a person does. Read the trajectory - if the run was close "
        "to converging, the ceiling is set too low; if it was thrashing, this "
        "saved the spend."
    ),
    "step_budget_exhausted": (
        "The global backstop fired, which means a specific budget failed to catch "
        "a cycle. Treat as a routing bug and read the trajectory."
    ),
}


@dataclass(frozen=True)
class Handover:
    """Everything a person needs, without re-running the graph."""

    run_id: str
    question: str
    trigger: str
    reason: str
    next_step: str
    draft: str | None
    violations: list[dict[str, Any]]
    evidence_ids: list[str]
    queries_tried: list[str]
    trajectory: list[dict[str, Any]]

    def render(self) -> str:
        lines = [
            f"ESCALATED: {self.question}",
            f"  run      {self.run_id}",
            f"  trigger  {self.trigger}",
            f"  reason   {self.reason}",
            f"  next     {self.next_step}",
            f"  queries  {self.queries_tried}",
            f"  evidence {len(self.evidence_ids)} chunks",
        ]
        if self.violations:
            lines.append("  unresolved violations:")
            lines += [f"    [{v['kind']}] {v['claim']}" for v in self.violations]
        lines.append("  trajectory:")
        lines += [f"    {s['step']}. {s['route']:<12} {s['trigger']}" for s in self.trajectory]
        return "\n".join(lines)


def flag_for_human(state: AgentState) -> Handover:
    """Build the handover. Terminal — returns a report, not a new state."""
    last = state.decisions[-1] if state.decisions else None
    trigger = last.trigger if last else "unknown"
    with span("node.flag_for_human", KIND_AGENT, trigger=trigger):
        return Handover(
            run_id=state.run_id,
            question=state.question,
            trigger=trigger,
            reason=last.reason if last else "no routing decision recorded",
            next_step=NEXT_STEPS.get(
                trigger, "Unrecognised trigger — read the trajectory below."
            ),
            draft=state.draft.text if state.draft else None,
            violations=[
                {"kind": v.kind, "claim": v.claim, "explanation": v.explanation}
                for v in (state.faithfulness.violations if state.faithfulness else [])
            ],
            evidence_ids=sorted(state.evidence_ids),
            queries_tried=list(state.queries_issued),
            trajectory=state.trajectory(),
        )
