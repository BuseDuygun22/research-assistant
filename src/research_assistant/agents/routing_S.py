"""Deterministic routing policy for the editor's outgoing edges (Sude).

The division of labour here is the point. The *model* classifies — it decides
that a sentence is an `unsupported_claim` rather than an `overstated_certainty`.
Application logic decides what to *do* about that classification. Letting the
model pick the next node as well would make the control flow a sampled variable,
and a control flow that varies run to run cannot be tested, budgeted, or
reproduced in a CI gate.

Everything in this module is a pure function of (verdicts, budget). No I/O, no
model calls, no clock. That is what makes the edge cases below testable at all.

Routing lives here rather than on the verdict contracts because it needs budget
state, and budget is an orchestration concern the contracts know nothing about.
The contracts own the taxonomy; this owns the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from research_assistant.contracts.judge_J import (
    AnswerVerdict,
    EvidenceAssessment,
    FaithfulnessVerdict,
)

Route = Literal["accept", "rewrite", "re_retrieve", "escalate", "abstain"]

# Machine-readable reason codes. Logged on every decision so an unexpected route
# can be explained after the fact without re-running the graph.
Trigger = Literal[
    "passed",
    "unclassified_violation",
    "unexplained_failure",
    "low_faithfulness_confidence",
    "low_answer_confidence",
    "evidence_insufficient",
    "deep_violation",
    "shallow_violation",
    "answer_incomplete",
    "rewrite_budget_exhausted",
    "re_retrieval_budget_exhausted",
    "step_budget_exhausted",
    "cost_ceiling",
    "evidence_unusable",
    "evidence_ok",
    "evidence_conflicting",
]

NON_TERMINAL_TRIGGERS: frozenset[str] = frozenset(
    {
        "passed",
        "deep_violation",
        "shallow_violation",
        "answer_incomplete",
        "evidence_ok",
        "evidence_conflicting",
        "evidence_unusable",
    }
)
"""Triggers that continue the loop rather than ending it.

Every *other* trigger reaches `flag_for_human`, and each of those must carry
guidance in `nodes/flag_for_human_S.NEXT_STEPS` — an escalation that cannot say
what a human should do next discards the diagnosis the system already made.
The set is named here rather than inferred so the check is a lookup rather than
a guess about control flow.
"""


# --- cost ---------------------------------------------------------------------
#
# Relative cost units, not currency. What matters for routing is the *ratio*
# between operations, which is stable across providers; absolute prices are not.
# Anchored at 1.0 = one short generation call.
#
# These are [D] estimates, and they are here rather than scattered as intuitions
# so they can be replaced with measurements from the tracing spans once the
# system runs against a real backend.
ROUTE_COST: dict[str, float] = {
    "accept": 0.0,
    "write": 0.0,  # the first draft is the baseline, not a revision
    "abstain": 0.0,
    "escalate": 0.0,  # cheap for us, expensive for a person — see HUMAN_COST
    "rewrite": 1.0,  # one writer call
    # Retrieval round + cross-encoder rerank over the candidate pool + triage
    # call + the writer call that follows. The reranker dominates.
    "re_retrieve": 4.0,
}

HUMAN_COST = 50.0
"""What an escalation costs in the same units.

Deliberately large. Escalation is cheap in machine time and expensive in the one
resource the project actually has least of, and a cost model that ignores that
would happily escalate everything. It is the reason `should_escalate_early`
exists: escalating *sooner* is only worth it when the remaining automated budget
is genuinely unlikely to succeed."""


def route_cost(route: Route) -> float:
    return ROUTE_COST.get(route, 1.0)


@dataclass(frozen=True)
class Budget:
    """Separate counters, because the two revision types are not interchangeable.

    A rewrite costs one writer call. A re-retrieval costs a retrieval round, a
    rerank over the candidate pool, *and* a subsequent writer call. Charging them
    against one shared counter means three cheap rewrites can starve the one
    expensive re-retrieval that would actually have fixed the draft — or the
    reverse. `steps_used` is the backstop against a cycle neither specific
    counter catches.
    """

    rewrites_used: int = 0
    re_retrievals_used: int = 0
    steps_used: int = 0
    cost_spent: float = 0.0
    max_rewrites: int = 3
    max_re_retrievals: int = 2
    max_steps: int = 12

    def spend(self, route: Route) -> Budget:
        """Return the budget after taking `route`. Frozen, so callers cannot
        mutate a shared budget by accident."""
        return Budget(
            rewrites_used=self.rewrites_used + (1 if route == "rewrite" else 0),
            re_retrievals_used=self.re_retrievals_used + (1 if route == "re_retrieve" else 0),
            steps_used=self.steps_used + 1,
            cost_spent=self.cost_spent + route_cost(route),
            max_rewrites=self.max_rewrites,
            max_re_retrievals=self.max_re_retrievals,
            max_steps=self.max_steps,
        )

    @property
    def rewrites_left(self) -> int:
        return max(0, self.max_rewrites - self.rewrites_used)

    @property
    def re_retrievals_left(self) -> int:
        return max(0, self.max_re_retrievals - self.re_retrievals_used)

    @property
    def steps_left(self) -> int:
        return max(0, self.max_steps - self.steps_used)

    @property
    def remaining_cost(self) -> float:
        """Cost of spending every budget that is left.

        Used to decide whether continuing is worth it: when the most the
        automated path could still cost approaches what a human costs, the
        cheap move is to stop pretending and escalate now.
        """
        return self.rewrites_left * route_cost("rewrite") + (
            self.re_retrievals_left * route_cost("re_retrieve")
        )

    def should_escalate_early(self, *, human_cost: float = HUMAN_COST) -> bool:
        """True when finishing the budget costs more than a person would.

        Guards the case the per-type caps miss: a run that has burned most of
        both budgets and is still failing is unlikely to be rescued by the
        remainder, and spending it is a worse use of the same resources than
        handing over now. Off by default (`cost_aware_escalation`) because it
        trades a small amount of autonomy for cost, and that is a call the
        operator should make rather than inherit.
        """
        return self.cost_spent + self.remaining_cost > human_cost


@dataclass(frozen=True)
class RoutingDecision:
    """A route plus why it was taken.

    `reason` exists because "the agent looped three times and gave up" is not a
    diagnosis. Carrying the trigger and the budget snapshot turns an odd
    trajectory into something readable in a trace without a re-run.
    """

    route: Route
    trigger: Trigger
    reason: str
    budget: Budget

    @property
    def node(self) -> str:
        """The graph node this route dispatches to."""
        return {
            "accept": "END",
            "rewrite": "writer",
            "re_retrieve": "researcher",
            "escalate": "flag_for_human",
            "abstain": "flag_for_human",
        }[self.route]


def decide_route(
    faithfulness: FaithfulnessVerdict,
    *,
    budget: Budget,
    answer: AnswerVerdict | None = None,
    min_faithfulness_confidence: float = 0.0,
    min_answer_confidence: float = 0.0,
    cost_aware: bool = False,
) -> RoutingDecision:
    """Pick the editor's outgoing edge.

    Order is load-bearing and each position is defended:

    1. **Step budget** — a global backstop before anything else, so a bug in the
       rules below cannot produce an unbounded loop.
    2. **Unclassified violations** — fail closed. A violation kind with no depth
       assignment routes to a human, never to a default branch. The alternative
       (treat unknown as shallow) sends evidence problems to the writer, which is
       the precise failure the depth split exists to prevent.
    3. **Uncertainty** — checked before `passed`, because a pass the judge is
       unsure about and a confident pass are different events and must not take
       the same edge.
    4. **Abstention** — if the evidence cannot answer the question, neither
       rewriting nor re-retrieving helps; more retrieval over a corpus that lacks
       the answer just produces a more confident wrong answer.
    5. **Faithfulness** — grounded before useful. An ungrounded draft is repaired
       before anyone asks whether it was on topic, because judging the relevance
       of unsupported prose is meaningless.
    6. **Answer relevance** — last, and only once the draft is grounded. This is
       what stops the gate rewarding a timid, perfectly-cited non-answer.
    """
    # 1. Global backstop.
    if budget.steps_used >= budget.max_steps:
        return RoutingDecision(
            "escalate",
            "step_budget_exhausted",
            f"step budget exhausted ({budget.steps_used}/{budget.max_steps}); "
            "escalating rather than continuing",
            budget,
        )

    # 1b. Cost ceiling. Off by default: it trades autonomy for spend, and that
    # is an operator decision rather than one to inherit. When on, it catches
    # the case the per-type caps miss - a run that has burned most of both
    # budgets and is still failing is unlikely to be rescued by the remainder.
    if cost_aware and budget.should_escalate_early():
        return RoutingDecision(
            "escalate",
            "cost_ceiling",
            f"spent {budget.cost_spent:.0f} units with {budget.remaining_cost:.0f} "
            f"still budgeted, against a human cost of {HUMAN_COST:.0f}; handing over "
            "now is cheaper than finishing the budget",
            budget,
        )

    # 2. Fail closed on anything we do not recognise.
    unclassified = faithfulness.unclassified_violations
    if unclassified:
        kinds = sorted({v.kind for v in unclassified})
        return RoutingDecision(
            "escalate",
            "unclassified_violation",
            f"violation kinds with no depth assignment: {kinds}; escalating rather "
            "than guessing a route",
            budget,
        )

    # 3. Uncertainty escalates before the verdict is trusted.
    if faithfulness.confidence < min_faithfulness_confidence:
        return RoutingDecision(
            "escalate",
            "low_faithfulness_confidence",
            f"faithfulness confidence {faithfulness.confidence:.2f} below threshold "
            f"{min_faithfulness_confidence:.2f}; verdict not trusted either way",
            budget,
        )
    if answer is not None and answer.confidence < min_answer_confidence:
        return RoutingDecision(
            "escalate",
            "low_answer_confidence",
            f"answer-relevance confidence {answer.confidence:.2f} below threshold "
            f"{min_answer_confidence:.2f}",
            budget,
        )

    # 4. The corpus cannot answer this. Abstain rather than retrieve harder.
    if answer is not None and answer.should_abstain:
        return RoutingDecision(
            "abstain",
            "evidence_insufficient",
            "retrieved evidence cannot answer the question; abstaining rather than "
            "summarising the nearest available material",
            budget,
        )

    # 5. Grounding problems first.
    if not faithfulness.passed:
        if not faithfulness.violations:
            # The rubric failed the draft without itemising why. Sending this to
            # the writer would hand it "fix it" with nothing to fix — the
            # unguided-revision case the self-correction literature says makes
            # output worse. Escalate: a rubric that can fail without a reason is
            # a rubric bug, and a human should see it.
            return RoutingDecision(
                "escalate",
                "unexplained_failure",
                "verdict is passed=False with no itemised violations; nothing to "
                "route on and nothing for the writer to repair",
                budget,
            )
        if faithfulness.deep_violations:
            if budget.re_retrievals_left == 0:
                return RoutingDecision(
                    "escalate",
                    "re_retrieval_budget_exhausted",
                    f"{len(faithfulness.deep_violations)} deep violation(s) remain but "
                    f"re-retrieval budget is spent "
                    f"({budget.re_retrievals_used}/{budget.max_re_retrievals}); "
                    "escalating rather than asking the writer to repair evidence it "
                    "does not have",
                    budget,
                )
            kinds = sorted({v.kind for v in faithfulness.deep_violations})
            return RoutingDecision(
                "re_retrieve",
                "deep_violation",
                f"deep violation(s) {kinds} need evidence not in context",
                budget,
            )
        if budget.rewrites_left == 0:
            return RoutingDecision(
                "escalate",
                "rewrite_budget_exhausted",
                f"shallow violations remain but rewrite budget is spent "
                f"({budget.rewrites_used}/{budget.max_rewrites})",
                budget,
            )
        kinds = sorted({v.kind for v in faithfulness.shallow_violations})
        return RoutingDecision(
            "rewrite",
            "shallow_violation",
            f"shallow violation(s) {kinds} are repairable from evidence already in "
            "context",
            budget,
        )

    # 6. Grounded. Is it actually an answer?
    if answer is not None and not answer.answers_question:
        if budget.re_retrievals_left == 0:
            return RoutingDecision(
                "escalate",
                "re_retrieval_budget_exhausted",
                f"draft is grounded but scores {answer.relevance}/3 on answering the "
                "question, and re-retrieval budget is spent",
                budget,
            )
        missing = answer.missing_aspects or ["(unspecified)"]
        return RoutingDecision(
            "re_retrieve",
            "answer_incomplete",
            f"draft is faithful but scores {answer.relevance}/3 on answering the "
            f"question; missing: {missing}",
            budget,
        )

    return RoutingDecision(
        "accept",
        "passed",
        "grounded and answers the question",
        budget,
    )


# --- pre-generation triage ----------------------------------------------------

EvidenceRoute = Literal["write", "re_retrieve", "abstain", "escalate"]


@dataclass(frozen=True)
class EvidenceDecision:
    """What to do with a retrieved set, decided before the writer runs."""

    route: EvidenceRoute
    trigger: Trigger
    reason: str
    budget: Budget

    @property
    def node(self) -> str:
        return {
            "write": "writer",
            "re_retrieve": "researcher",
            "abstain": "flag_for_human",
            "escalate": "flag_for_human",
        }[self.route]


def decide_evidence_route(
    assessment: EvidenceAssessment,
    *,
    budget: Budget,
    min_confidence: float = 0.0,
) -> EvidenceDecision:
    """Route on the retrieved set, before spending a writer call on it.

    The ordering question that matters here is what to do about `insufficient`
    when there is no retrieval budget left. The answer is **abstain**, not
    escalate: "the corpus does not contain an answer" is a correct, useful
    output that a human cannot improve on by looking at the same passages. An
    escalation would hand a person the same evidence and the same conclusion.

    A low-confidence assessment is *not* escalated. Triage is an optimisation —
    it saves a wasted writer call — and an unsure triage should fall through to
    the full draft-and-judge path rather than short-circuit it. Escalating on it
    would make the cheap check the thing that stops the run.
    """
    if budget.steps_used >= budget.max_steps:
        return EvidenceDecision(
            "escalate",
            "step_budget_exhausted",
            f"step budget exhausted ({budget.steps_used}/{budget.max_steps}) before "
            "evidence could be assessed",
            budget,
        )

    if assessment.confidence < min_confidence:
        return EvidenceDecision(
            "write",
            "evidence_ok",
            f"triage confidence {assessment.confidence:.2f} below threshold "
            f"{min_confidence:.2f}; falling through to the full draft-and-judge path "
            "rather than trusting a cheap check to stop the run",
            budget,
        )

    if assessment.label == "conflicting":
        kinds = sorted({c.kind for c in assessment.conflicts})
        return EvidenceDecision(
            "write",
            "evidence_conflicting",
            f"{len(assessment.conflicts)} conflict(s) {kinds} in the retrieved set; "
            "writing a draft that reports the disagreement rather than resolving it",
            budget,
        )

    if assessment.label == "insufficient":
        if budget.re_retrievals_left > 0:
            return EvidenceDecision(
                "re_retrieve",
                "evidence_unusable",
                f"retrieved passages cannot answer the question ({assessment.rationale}); "
                "retrieving again before spending a writer call on them",
                budget,
            )
        return EvidenceDecision(
            "abstain",
            "evidence_insufficient",
            "retrieved passages cannot answer the question and the retrieval budget "
            "is spent; abstaining rather than escalating, because a human given the "
            "same passages would reach the same conclusion",
            budget,
        )

    return EvidenceDecision(
        "write",
        "evidence_ok",
        f"evidence is {assessment.label}; proceeding to draft",
        budget,
    )
