"""Evidence triage node (Sude).

Sits between the researcher and the writer. Judges the retrieved set before a
writer call is spent on it, and routes: write, retrieve again, or abstain.

Writes `assessment` as the editor does. The ownership rule is about which kind of
work produces a field, and this is judgement rather than retrieval.
"""

from __future__ import annotations

from research_assistant.judge.triage_S import assess_evidence
from research_assistant.observability.tracing_S import KIND_AGENT, span

from ..routing_S import decide_evidence_route
from ..state_S import AgentState


def triage(state: AgentState) -> AgentState:
    """Assess the current evidence and record the evidence routing decision."""
    with span("node.triage", KIND_AGENT, n_evidence=len(state.evidence)) as sp:
        assessment = assess_evidence(state.question, state.evidence)
        state = state.apply("editor", assessment=assessment)
        decision = decide_evidence_route(assessment, budget=state.budget)
        sp.set(label=assessment.label, route=decision.route, trigger=decision.trigger)
        return state.record(decision)  # type: ignore[arg-type]
