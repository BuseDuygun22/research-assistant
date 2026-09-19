"""Editor node (Sude).

Runs the judges and applies the routing rule. It makes no judgement of its own —
it collects verdicts and calls `decide_route`, which is a pure function. That
split is the point: the model classifies, application logic decides control flow.
A sampled control flow cannot be budgeted, tested, or reproduced in a CI gate.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from research_assistant.config_J import get_settings
from research_assistant.judge.faithfulness_S import check_answer, check_faithfulness
from research_assistant.llm_S import LLMError
from research_assistant.observability.tracing_S import KIND_AGENT, span

from ..routing_S import RoutingDecision, decide_route
from ..state_S import AgentState

logger = logging.getLogger(__name__)


def editor(state: AgentState) -> AgentState:
    """Judge the current draft and record the routing decision it implies."""
    settings = get_settings()
    draft = state.draft
    if draft is None:
        raise RuntimeError("editor ran before a draft existed")

    with span("node.editor", KIND_AGENT, draft_id=draft.draft_id) as sp:
        try:
            faithfulness = check_faithfulness(draft.draft_id, draft.text, state.evidence)
            answer = check_answer(draft.draft_id, state.question, draft.text, state.evidence)
        except (LLMError, ValueError) as exc:
            # A verdict we could not parse is neither a pass nor a fail.
            # Escalating is the only honest response — routing on a verdict we do
            # not have would be guessing, and guessing here ships a draft.
            logger.exception("judge failed; escalating rather than guessing a route")
            return state.record(
                RoutingDecision(
                    "escalate",
                    "unexplained_failure",
                    f"judge did not return a usable verdict ({type(exc).__name__}); "
                    "escalating rather than routing on a verdict we do not have",
                    state.budget,
                )
            )

        state = state.apply("editor", faithfulness=faithfulness, answer=answer)

        # Mark which claims survived, so the next revision can keep them rather
        # than regenerate them. `claims` is owned by the writer, so the write is
        # made as the writer — the ownership rule is about which *field* changed,
        # not which node happened to compute the value.
        if draft.claims:
            broken = {v.claim for v in faithfulness.violations}
            state = state.apply(
                "writer",
                claims=[replace(c, verified=c.text not in broken) for c in draft.claims],
            )

        decision = decide_route(
            faithfulness,
            budget=state.budget,
            answer=answer,
            min_faithfulness_confidence=settings.escalation_confidence,
            min_answer_confidence=settings.escalation_confidence,
        )
        sp.set(route=decision.route, trigger=decision.trigger)
        logger.info(
            "route=%s trigger=%s reason=%s", decision.route, decision.trigger, decision.reason
        )
        return state.record(decision)
