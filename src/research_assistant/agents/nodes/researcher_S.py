"""Researcher node (Sude).

Acquires evidence. Never writes prose — that separation is what keeps the writer
from quietly inventing support when retrieval comes up short.

Two behaviours worth naming:

**Reformulation is aimed, not repeated.** On a second round the node is told what
was already tried and what specifically is missing (from `retrieval_brief()`), so
a round costs something. A researcher that reissues a near-identical query burns
a re-retrieval budget — the most expensive budget we have — for nothing.

**Novelty is enforced, not requested.** The prompt asks for a different query; a
check verifies it. Asking a model not to repeat itself is a hope, and this loop
is expensive enough to want a guarantee.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, ConfigDict, Field

from research_assistant.config_J import get_settings
from research_assistant.contracts.mcp_tools_J import SearchPapersInput
from research_assistant.llm_S import LLMError, get_llm
from research_assistant.mcp_server.api_S import search_papers
from research_assistant.observability.tracing_S import KIND_AGENT, span

from ..state_S import AgentState

logger = logging.getLogger(__name__)

REFORMULATE_SYSTEM = (
    "You rewrite a research question into a search query for a corpus of "
    "academic papers.\n\n"
    "Temperature is high here on purpose: variety in vocabulary is what rescues "
    "a failed retrieval, and your output is a *query*, never text that reaches "
    "the draft, so exploration is safe.\n\n"
    "If previous queries are listed, your query MUST differ from all of them in "
    "substance, not only wording — try method names, author conventions, or the "
    "vocabulary a paper would use rather than the vocabulary the question uses.\n"
    "If missing aspects are listed, aim at those specifically.\n\n"
    "Return a single JSON object matching the schema."
)


class _Reformulation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    rationale: str = ""


def _normalise(q: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", q.lower()))


def is_novel(query: str, previous: list[str], *, max_overlap: float = 0.8) -> bool:
    """Is this query meaningfully different from what has already been tried?

    Jaccard over token sets. Crude on purpose — the job is catching a near-repeat,
    not semantic equivalence, and a cheap check that runs every round beats a
    good one that needs an embedding call.
    """
    tokens = _normalise(query)
    if not tokens:
        return False
    for old in previous:
        old_tokens = _normalise(old)
        union = tokens | old_tokens
        if union and len(tokens & old_tokens) / len(union) >= max_overlap:
            return False
    return True


def _fingerprint(text: str) -> str:
    """Normalised text identity, so a preprint and its published version count once."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def reformulate(state: AgentState) -> str:
    """Produce the next query. Falls back to the raw question on the first round."""
    brief = state.retrieval_brief()
    if not brief["already_tried"]:
        return state.question

    user = (
        f"QUESTION:\n{state.question}\n\n"
        f"ALREADY TRIED (do not repeat):\n" + "\n".join(f"- {q}" for q in brief["already_tried"])
    )
    if brief["missing_aspects"]:
        user += "\n\nMISSING ASPECTS:\n" + "\n".join(f"- {a}" for a in brief["missing_aspects"])
    if brief["unsupported_claims"]:
        user += "\n\nCLAIMS LACKING EVIDENCE:\n" + "\n".join(
            f"- {c}" for c in brief["unsupported_claims"]
        )

    for attempt in range(2):
        try:
            reply = get_llm().complete_json(
                system=REFORMULATE_SYSTEM, user=user, schema=_Reformulation
            )
        except LLMError:
            logger.warning("reformulation failed; falling back to the raw question")
            break
        if is_novel(reply.query, brief["already_tried"]):
            return reply.query
        logger.info("reformulation %d repeated a previous query; retrying", attempt + 1)
        user += f"\n\nREJECTED (too similar to a previous query): {reply.query}"

    # Both attempts repeated themselves. Widening beats looping: strip the
    # filters implied by the original phrasing rather than spending another round
    # on the same ground.
    return state.question


def researcher(state: AgentState) -> AgentState:
    """Retrieve evidence for the question, or for what the editor said is missing."""
    settings = get_settings()
    with span("node.researcher", KIND_AGENT, round=len(state.queries_issued)) as sp:
        query = reformulate(state)
        result = search_papers(
            SearchPapersInput(query=query, top_k=settings.writer_top_k, use_reranker=True)
        )
        sp.set(query=query, n_results=len(result.results))

        if result.error and not result.results:
            logger.warning("retrieval returned no usable results: %s", result.error.message)

        # Merge rather than replace: evidence from an earlier round may still be
        # supporting verified claims in the current draft, and dropping it would
        # turn those into unsupported claims on the next check.
        #
        # Deduplicate by normalised text as well as by id. A preprint and its
        # published version have different ids and identical text; kept as two
        # passages they read to the writer - and to the triage judge - as two
        # independent sources agreeing, which is corroboration that does not exist.
        existing = {r.chunk.chunk_id: r for r in state.evidence}
        seen_text = {_fingerprint(r.chunk.text) for r in state.evidence}
        for r in result.results:
            fp = _fingerprint(r.chunk.text)
            if r.chunk.chunk_id in existing or fp in seen_text:
                continue
            existing[r.chunk.chunk_id] = r
            seen_text.add(fp)

        return state.apply(
            "researcher",
            evidence=tuple(existing.values()),
            queries_issued=(*state.queries_issued, query),
        )
