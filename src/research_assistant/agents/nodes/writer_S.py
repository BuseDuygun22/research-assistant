"""Writer node (Sude).

Turns evidence into cited prose. It has no retrieval tool, deliberately: a writer
that can fetch is a writer that can paper over a retrieval failure, and the whole
routing design depends on evidence gaps surfacing as violations rather than being
quietly filled in.

On revision it receives `repair_brief()` — the specific failed claims and their
diagnoses — not "try again". Handing back a whole draft with a pass/fail is the
unguided-revision case the self-correction literature says makes output worse.
Verified claims are listed as keep-as-is, so correct prose survives untouched
rather than being regenerated and given a fresh chance to break.
"""

from __future__ import annotations

import hashlib
import logging

from pydantic import BaseModel, ConfigDict, Field

from research_assistant.config_J import get_settings
from research_assistant.judge.triage_S import describe_conflicts
from research_assistant.llm_S import LLMError, get_llm
from research_assistant.observability.tracing_S import KIND_AGENT, span
from research_assistant.prompts_S import with_examples

from ..routing_S import RoutingDecision
from ..state_S import AgentState, ClaimSpan, Draft

logger = logging.getLogger(__name__)

WRITE_SYSTEM = (
    "You write a short, cited answer to a research question using ONLY the "
    "passages provided.\n\n"
    "Rules:\n"
    "1. Every factual claim carries a citation in square brackets: [chunk_id].\n"
    "2. If the passages do not support a claim, do not make it. Saying the "
    "evidence does not cover something is a correct and expected answer, not a "
    "failure.\n"
    "3. Prefer the source's own hedging. If it says 'suggests', do not write "
    "'shows'.\n"
    "4. Split your answer into atomic claims in the `claims` field: one "
    "assertion each, with the chunk_id supporting it. The `text` of each claim "
    "must appear verbatim in `draft`.\n\n"
    "Return a single JSON object matching the schema."
)

REVISE_SYSTEM = (
    "You repair specific claims in an existing draft.\n\n"
    "You are given claims that failed verification, each with a diagnosis, and a "
    "list of claim ids to KEEP EXACTLY AS THEY ARE.\n\n"
    "Rules:\n"
    "1. Do not rewrite kept claims. They passed; regenerating them risks breaking "
    "what already works.\n"
    "2. Repair each failed claim according to its diagnosis:\n"
    "   - missing_citation: attach the chunk_id that supports it\n"
    "   - misattributed_citation: swap to the correct chunk_id\n"
    "   - overstated_certainty: weaken the claim to match the source's hedging\n"
    "3. If a claim cannot be repaired from the passages provided, DELETE it. Do "
    "not invent support. A shorter honest answer is the correct outcome.\n\n"
    "Return a single JSON object matching the schema."
)


class _ClaimOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    chunk_id: str | None = None


class _DraftOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft: str = Field(..., min_length=1)
    claims: list[_ClaimOut] = Field(default_factory=list)


def _draft_id(text: str, revision: int) -> str:
    return f"d{revision}-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def context_window(state: AgentState) -> str:
    """The writer's evidence window.

    Capped at `writer_top_k` spans. This is the context-rot budget: accuracy
    degrades with input length, worst in the middle, and semantically similar
    distractors — exactly what a single-topic paper corpus yields — make it worse.
    Retrieve wide, deliver narrow.
    """
    k = get_settings().writer_top_k
    return "\n\n".join(
        f"[{r.chunk.chunk_id}] ({r.chunk.metadata.title}, {r.chunk.metadata.section or 'n/a'})\n"
        f"{r.chunk.text}"
        for r in state.evidence[:k]
    )


def writer(state: AgentState) -> AgentState:
    """Write a first draft, or repair the current one."""
    current = state.draft
    revising = current is not None and state.faithfulness is not None
    with span("node.writer", KIND_AGENT, revising=revising) as sp:
        if current is not None and state.faithfulness is not None:
            brief = state.repair_brief()
            fixes = "\n".join(
                f"- [{c['kind']}] {c['claim']}\n  diagnosis: {c['explanation']}"
                for c in brief["claims_to_fix"]
            )
            user = (
                f"QUESTION:\n{state.question}\n\n"
                f"PASSAGES:\n{context_window(state)}\n\n"
                f"CURRENT DRAFT:\n{current.text}\n\n"
                f"CLAIMS TO FIX:\n{fixes}\n\n"
                f"KEEP EXACTLY: {brief['keep']}"
            )
            system = with_examples(REVISE_SYSTEM, "revise")
        else:
            user = f"QUESTION:\n{state.question}\n\nPASSAGES:\n{context_window(state)}"
            # Conflicts found at triage travel with the prompt. Left to itself a
            # model silently picks one side, and the draft is then perfectly
            # faithful to the passage it picked.
            if state.assessment is not None and state.assessment.has_conflicts:
                user += "\n\nCONFLICTS:\n" + describe_conflicts(state.assessment)
            system = with_examples(WRITE_SYSTEM, "write")

        try:
            out = get_llm().complete_json(system=system, user=user, schema=_DraftOut)
        except (LLMError, ValueError) as exc:
            # A network error, an expired key, a rate limit, or a retired model id
            # - none of that is a statement about the evidence or the question, and
            # must not be allowed to crash the whole run. Escalating (not
            # abstaining) keeps that distinction: abstain means "the corpus does
            # not have this", which is not what happened here.
            logger.exception("writer LLM call failed; escalating rather than crashing")
            return state.record(
                RoutingDecision(
                    "escalate",
                    "llm_backend_unavailable",
                    f"the configured LLM backend failed while writing a draft "
                    f"({type(exc).__name__}: {exc})",
                    state.budget,
                )
            )
        revision = len(state.drafts)
        draft_id = _draft_id(out.draft, revision)
        sp.set(draft_id=draft_id, n_claims=len(out.claims))

        claims = tuple(
            ClaimSpan(claim_id=f"{draft_id}-c{i}", text=c.text, chunk_id=c.chunk_id)
            for i, c in enumerate(out.claims)
        )
        return state.add_draft(
            Draft(
                draft_id=draft_id,
                text=out.draft,
                claims=claims,
                revision_of=state.draft.draft_id if state.draft else None,
            )
        )
