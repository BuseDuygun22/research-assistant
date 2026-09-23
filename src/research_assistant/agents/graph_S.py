"""The agent graph (Sude).

researcher → triage → writer → editor → {accept | rewrite | re_retrieve | abstain | escalate}
                   └→ re_retrieve | abstain   (decided before any writer call)

Two runners, one behaviour:

* `run` — a plain deterministic loop. No framework, no import of langgraph. This
  is the one the eval gate and the tests use, because a gate that depends on an
  optional package is a gate that breaks on the push where it matters.
* `build_langgraph` — wires the same node functions into a `StateGraph` for
  anyone who wants LangGraph's tracing and streaming.

The important property is that both call the *same* `decide_route`. The framework
is presentation; the policy is one pure function, tested independently of either
runner. Swapping frameworks cannot change what the system decides.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from research_assistant.config_J import get_settings
from research_assistant.observability.tracing_S import KIND_AGENT, flush, span

from .nodes.editor_S import editor
from .nodes.flag_for_human_S import Handover, flag_for_human
from .nodes.researcher_S import researcher
from .nodes.triage_S import triage
from .nodes.writer_S import writer
from .routing_S import Budget
from .state_S import AgentState

logger = logging.getLogger(__name__)

TERMINAL = {"accept", "abstain", "escalate"}


def initial_state(question: str, *, run_id: str | None = None) -> AgentState:
    s = get_settings()
    return AgentState(
        run_id=run_id or uuid.uuid4().hex[:12],
        question=question,
        budget=Budget(
            max_rewrites=s.max_rewrites,
            max_re_retrievals=s.max_re_retrievals,
            # Forced to 0 when the feature is off, so a nonzero value left over
            # in config cannot enable live discovery by accident.
            max_live_discoveries=s.max_live_discoveries if s.allow_live_discovery else 0,
            max_steps=s.max_steps,
        ),
    )


def gather(state: AgentState) -> AgentState:
    """Retrieve, then triage — retrying retrieval (local re-query, then, if
    enabled and unspent, one live discovery attempt) while triage says the
    evidence is unusable and budget remains.

    Returns when the evidence is worth writing from, or when the run has ended
    (abstain / escalate); callers check `is_terminal`. Bounded twice: by the
    retrieval budget inside `decide_evidence_route`, and by this loop's ceiling.
    """
    state = researcher(state)
    for _ in range(state.budget.max_steps + 1):
        state = triage(state)
        if state.decisions[-1].route not in ("re_retrieve", "discover"):
            return state
        state = researcher(state)
    return state


def run(question: str, *, run_id: str | None = None) -> tuple[AgentState, Handover | None]:
    """Run one question to a terminal state.

    Returns the final state and, when the run escalated or abstained, the
    handover a person would read. Both are returned rather than one-or-the-other
    so a caller can always inspect the trajectory.

    The loop bound is `max_steps + 1` — a hard ceiling independent of the routing
    rules, so a bug in `decide_route` produces a caught, logged stop rather than a
    hung process. The router already guards this; the runner does not trust it to.
    """
    state = initial_state(question, run_id=run_id)
    with span("graph.run", KIND_AGENT, question=question[:120]) as sp:
        state = gather(state)
        if not state.is_terminal:
            state = writer(state)

        for _ in range(state.budget.max_steps + 1):
            if state.is_terminal:
                break
            state = editor(state)
            decision = state.decisions[-1]
            if decision.route in TERMINAL:
                break
            if decision.route == "rewrite":
                state = writer(state)
            elif decision.route in ("re_retrieve", "discover"):
                state = gather(state)
                if state.is_terminal:
                    break
                state = writer(state)
        else:
            logger.error(
                "runner ceiling hit for run %s — decide_route failed to terminate", state.run_id
            )

        sp.set(outcome=state.outcome, steps=len(state.decisions))
        handover = flag_for_human(state) if state.outcome in ("escalated", "abstained") else None
        flush()
        return state, handover


# --- LangGraph adapter -------------------------------------------------------


def build_langgraph() -> Any:
    """Wire the same nodes into a LangGraph `StateGraph`.

    Imported lazily so this module loads without langgraph installed — the gate
    and the unit tests use `run` above and must not acquire an optional
    dependency by importing this file.
    """
    from langgraph.graph import END, StateGraph  # noqa: PLC0415

    def _route(state: AgentState) -> str:
        decision = state.decisions[-1]
        return {
            "write": "writer",
            "accept": END,
            "abstain": "flag_for_human",
            "escalate": "flag_for_human",
            "rewrite": "writer",
            "re_retrieve": "researcher",
        }[decision.route]

    graph = StateGraph(AgentState)
    graph.add_node("researcher", researcher)
    graph.add_node("triage", triage)
    graph.add_node("writer", writer)
    graph.add_node("editor", editor)
    graph.add_node("flag_for_human", lambda s: s)  # terminal; Handover built by the caller
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "triage")
    graph.add_conditional_edges("triage", _route)
    graph.add_edge("writer", "editor")
    graph.add_conditional_edges("editor", _route)
    graph.add_edge("flag_for_human", END)
    return graph.compile()
