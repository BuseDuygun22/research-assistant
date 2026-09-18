"""FastAPI retrieval/citation/summary service (Sude).

Architecture (agreed): the MCP server is a thin protocol adapter; the tool
*logic* lives here, behind plain HTTP. The split buys three things —

1. the logic is testable with `TestClient`, without speaking MCP;
2. the eval gate and the agent graph can call it directly, so a CI run does not
   have to spawn an MCP session just to measure retrieval;
3. the MCP layer stays small enough that a protocol change is a one-file change.

Endpoints mirror the tool schemas in `contracts/mcp_tools_J.py` one-for-one. They
return `ToolError` in the body with HTTP 200 rather than raising, because the MCP
tools must surface a recoverable error to the agent, not a transport failure.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI

from research_assistant.config_J import get_settings
from research_assistant.contracts.mcp_tools_J import (
    GetCitationInput,
    GetCitationOutput,
    SearchPapersInput,
    SearchPapersOutput,
    SummarizeSectionInput,
    SummarizeSectionOutput,
    ToolError,
)
from research_assistant.contracts.retrieval_J import Chunk, RetrievalRequest
from research_assistant.llm_S import get_llm
from research_assistant.mcp_server.backend_S import get_backend
from research_assistant.mcp_server.deadline_S import ToolTimeout, with_deadline
from research_assistant.mcp_server.health_S import HealthStatus, liveness, readiness
from research_assistant.observability.tracing_S import KIND_TOOL, span

logger = logging.getLogger(__name__)

app = FastAPI(title="research-assistant retrieval API", version="0.1.0")


@app.get("/health", response_model=HealthStatus)
def health() -> HealthStatus:
    return liveness()


@app.get("/ready", response_model=HealthStatus)
def ready() -> HealthStatus:
    return readiness()


@app.post("/tools/search_papers", response_model=SearchPapersOutput)
def search_papers(payload: SearchPapersInput) -> SearchPapersOutput:
    with span("tool.search_papers", KIND_TOOL, top_k=payload.top_k) as sp:
        t0 = time.perf_counter()
        filters: dict = {}
        if payload.year_min is not None:
            filters["year_min"] = payload.year_min
        if payload.year_max is not None:
            filters["year_max"] = payload.year_max

        request = RetrievalRequest(
            query=payload.query,
            top_k=payload.top_k,
            # Over-fetch before reranking. 4x top_k is the usual
            # operating point: enough headroom for the reranker to move
            # a buried positive up, without paying for a long tail the
            # cross-encoder would have to score.
            candidate_k=min(200, max(20, payload.top_k * 4)),
            filters=filters,
            use_reranker=payload.use_reranker,
        )
        try:
            response = with_deadline(
                "search_papers",
                get_settings().tool_deadline_seconds,
                lambda: get_backend().retrieve(request),
            )
        except ToolTimeout as exc:
            # A hang is the one failure an agent cannot see: no error, no budget
            # spent, no escalation — it just stops. Turning it into an ordinary
            # retryable ToolError is what puts it back under the routing rules.
            logger.warning("search_papers timed out after %.1fs", exc.seconds)
            return SearchPapersOutput(
                corpus_version=get_settings().corpus_version,
                reranker_version="unknown",
                error=ToolError(
                    code="timeout",
                    message=str(exc),
                    retryable=True,
                    remediation="Retry with a smaller top_k; the candidate pool scales "
                    "with it and the reranker is the usual cause of a slow call.",
                ),
            )
        except Exception as exc:  # noqa: BLE001 - never cross the boundary as a raise
            logger.exception("search_papers failed")
            return SearchPapersOutput(
                corpus_version=get_settings().corpus_version,
                reranker_version="unknown",
                error=ToolError(
                    code="retrieval_failed",
                    message=str(exc),
                    retryable=True,
                    remediation="Retry once; if it fails again the index is down and "
                    "reformulating the query will not help.",
                ),
            )

        sp.set(n_results=len(response.results), ms=(time.perf_counter() - t0) * 1000)
        if not response.results:
            return SearchPapersOutput(
                corpus_version=response.corpus_version,
                reranker_version=response.reranker_version,
                error=ToolError(
                    code="no_results",
                    message="No chunk matched the query.",
                    retryable=True,
                    remediation=(
                        "Broaden the query: drop the year filter, or substitute "
                        "method names for general terms."
                        if (payload.year_min or payload.year_max)
                        else "Broaden the query with synonyms or a more general phrasing."
                    ),
                ),
            )
        return SearchPapersOutput(
            results=response.results,
            corpus_version=response.corpus_version,
            reranker_version=response.reranker_version,
        )


@app.post("/tools/get_citation", response_model=GetCitationOutput)
def get_citation(payload: GetCitationInput) -> GetCitationOutput:
    with span("tool.get_citation", KIND_TOOL, style=payload.style):
        chunk = get_backend().get_chunk(payload.chunk_id)
        if chunk is None:
            return GetCitationOutput(
                chunk_id=payload.chunk_id,
                error=ToolError(
                    code="chunk_not_found",
                    message="Unknown chunk_id. It must come from a search_papers result "
                    "in this same session - ids are corpus-version specific.",
                    retryable=False,
                    remediation="Call search_papers again and cite a chunk_id from the "
                    "fresh results; do not reuse ids from an earlier corpus version.",
                ),
            )
        return GetCitationOutput(
            chunk_id=chunk.chunk_id,
            formatted=format_citation(chunk, payload.style),
            quote=chunk.text.strip(),
            paper_id=chunk.metadata.paper_id,
            title=chunk.metadata.title,
            page=chunk.metadata.page,
        )


@app.post("/tools/summarize_section", response_model=SummarizeSectionOutput)
def summarize_section(payload: SummarizeSectionInput) -> SummarizeSectionOutput:
    with span("tool.summarize_section", KIND_TOOL, n_chunks=len(payload.chunk_ids)):
        backend = get_backend()
        found: list[Chunk] = []
        missing: list[str] = []
        for cid in payload.chunk_ids:
            chunk = backend.get_chunk(cid)
            if chunk is None:
                missing.append(cid)
            else:
                found.append(chunk)

        if not found:
            return SummarizeSectionOutput(
                error=ToolError(
                    code="chunk_not_found",
                    message=f"None of the requested chunk_ids exist: {missing}",
                    retryable=False,
                    remediation="Call search_papers to obtain valid chunk_ids for this "
                    "corpus version, then summarize those.",
                )
            )

        # Chunks are labelled in the prompt so the summary can be traced back to
        # a specific source, and so `grounded_in` is a claim we can check rather
        # than an assumption.
        body = "\n\n".join(
            f"[{c.chunk_id}] ({c.metadata.title}, {c.metadata.section or 'n/a'})\n{c.text}"
            for c in found
        )
        system = (
            "You summarise scientific text. Use only the provided excerpts. "
            "Do not add background knowledge. If the excerpts do not answer the "
            "focus question, say so explicitly rather than filling the gap."
        )
        user = (
            (f"Focus question: {payload.focus}\n\n" if payload.focus else "")
            + f"Excerpts:\n{body}\n\nWrite at most {payload.max_words} words."
        )
        try:
            summary = with_deadline(
                "summarize_section",
                get_settings().tool_deadline_seconds,
                lambda: get_llm().complete(system, user, max_tokens=payload.max_words * 2),
            )
        except ToolTimeout as exc:
            return SummarizeSectionOutput(
                error=ToolError(
                    code="timeout",
                    message=str(exc),
                    retryable=True,
                    remediation="Retry with fewer chunk_ids or a smaller max_words.",
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("summarize_section failed")
            return SummarizeSectionOutput(
                error=ToolError(
                    code="generation_failed",
                    message=str(exc),
                    retryable=True,
                    remediation="Retry once with fewer chunk_ids; the request may have "
                    "exceeded the generation budget.",
                )
            )

        # Some ids resolved and some did not. Previously the missing ones were
        # dropped silently, which let the writer cite a section it never saw
        # summarised. Report the gap as a partial error instead: the summary is
        # usable, but the agent is told what is absent from it.
        if missing:
            return SummarizeSectionOutput(
                summary=summary,
                grounded_in=[c.chunk_id for c in found],
                error=ToolError(
                    code="chunk_not_found",
                    message=f"Summarised {len(found)} of {len(payload.chunk_ids)} chunks; "
                    f"these ids do not exist: {missing}",
                    retryable=False,
                    partial=True,
                    remediation="Use the summary, but do not attribute claims to the "
                    "missing ids; re-run search_papers if that evidence is needed.",
                ),
            )

        return SummarizeSectionOutput(
            summary=summary,
            grounded_in=[c.chunk_id for c in found],
        )


def format_citation(chunk: Chunk, style: str) -> str:
    """Render a citation. Kept here rather than in the agent so every consumer
    produces byte-identical strings - citation precision is measured by string
    match against the draft, so two renderings of the same source would score as
    a miss."""
    m = chunk.metadata
    authors = m.authors or ["Anon."]
    first = authors[0].split()[-1] if authors[0] else "Anon."
    et_al = f"{first} et al." if len(authors) > 1 else first
    year = m.year or "n.d."

    if style == "inline":
        return f"({et_al}, {year})"
    if style == "bibtex":
        key = f"{first.lower()}{year}"
        return (
            f"@article{{{key},\n"
            f"  title  = {{{m.title}}},\n"
            f"  author = {{{' and '.join(authors)}}},\n"
            f"  year   = {{{year}}},\n"
            f"  note   = {{{m.venue or m.source_uri or m.paper_id}}}\n"
            f"}}"
        )
    # apa
    venue = f" {m.venue}." if m.venue else ""
    page = f" p. {m.page}." if m.page else ""
    return f"{', '.join(authors)} ({year}). {m.title}.{venue}{page}"
