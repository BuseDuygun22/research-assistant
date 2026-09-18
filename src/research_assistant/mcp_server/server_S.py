"""MCP protocol adapter (Sude).

Thin by design. Every tool here does three things: validate the input against the
shared contract, call the FastAPI handler that holds the logic, and serialise the
result. No business logic lives in this file, which is what makes a protocol
change a one-file change and lets the eval gate call the same logic without
spawning an MCP session per query.

Tool descriptions carry a worked example each. That is not decoration: concrete
examples measurably improve an agent's parameter handling, and three tools is few
enough that the schema cost is irrelevant. The reason to keep the tool count
small is the same one — schema bloat is a real context tax past a dozen or so.
"""

from __future__ import annotations

import json
import logging

from research_assistant.config_J import get_settings
from research_assistant.contracts.mcp_tools_J import (
    GetCitationInput,
    SearchPapersInput,
    SummarizeSectionInput,
)
from research_assistant.mcp_server.api_S import get_citation, search_papers, summarize_section
from research_assistant.mcp_server.health_S import readiness

logger = logging.getLogger(__name__)

SEARCH_DESCRIPTION = """Search the paper corpus for passages relevant to a question.

Returns ranked chunks with separate bm25/vector/rerank scores and a chunk_id you
pass to get_citation or summarize_section.

Example: {"query": "reciprocal rank fusion vs weighted score blending",
"top_k": 5, "year_min": 2009}

On no results you get error.code="no_results" with error.retryable=true and a
remediation telling you how to broaden. Follow the remediation rather than
reissuing the same query."""

CITATION_DESCRIPTION = """Get a formatted citation and the verbatim supporting span for a chunk.

The quote and its character offsets are what make a claim checkable — cite the
span, not the paper.

Example: {"chunk_id": "d781365c5b5cd64b", "style": "apa"}

chunk_ids are corpus-version specific. If you get error.code="chunk_not_found",
re-run search_papers; do not reuse ids from an earlier session."""

SUMMARIZE_DESCRIPTION = """Summarize one or more chunks, optionally toward a specific question.

Example: {"chunk_ids": ["d781365c5b5cd64b", "0f2e14c7e45065d3"],
"focus": "how the two fusion methods differ", "max_words": 200}

If some ids do not exist you still get a summary, plus an error with
partial=true naming the missing ids. Use the summary, but do not attribute
claims to the ids it names."""


def build_server():  # type: ignore[no-untyped-def]
    """Construct the MCP server. Imported lazily by callers so this module does
    not force the `mcp` dependency on the eval gate or the tests."""
    # mcp 2.x renamed FastMCP to MCPServer; the decorator API is otherwise the same.
    from mcp.server.mcpserver import MCPServer  # noqa: PLC0415

    mcp = MCPServer("research-assistant")

    # Explicit names: the default is the Python function name, which would
    # advertise "search_papers_tool" rather than the name the contracts define.
    @mcp.tool(name="search_papers", description=SEARCH_DESCRIPTION)
    def search_papers_tool(
        query: str,
        top_k: int = 5,
        year_min: int | None = None,
        year_max: int | None = None,
        use_reranker: bool = True,
    ) -> str:
        payload = SearchPapersInput(
            query=query,
            top_k=top_k,
            year_min=year_min,
            year_max=year_max,
            use_reranker=use_reranker,
        )
        return search_papers(payload).model_dump_json()

    @mcp.tool(name="get_citation", description=CITATION_DESCRIPTION)
    def get_citation_tool(chunk_id: str, style: str = "apa") -> str:
        return get_citation(GetCitationInput(chunk_id=chunk_id, style=style)).model_dump_json()

    @mcp.tool(name="summarize_section", description=SUMMARIZE_DESCRIPTION)
    def summarize_section_tool(
        chunk_ids: list[str], focus: str | None = None, max_words: int = 200
    ) -> str:
        payload = SummarizeSectionInput(
            chunk_ids=chunk_ids, focus=focus, max_words=max_words
        )
        return summarize_section(payload).model_dump_json()

    @mcp.resource("health://readiness", name="readiness", mime_type="application/json")
    def readiness_resource() -> str:
        """Exposed so a client can tell a degraded backend from a healthy one
        before trusting any numbers that come out of it."""
        return json.dumps(readiness().model_dump())

    return mcp


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    status = readiness()
    if status.status != "ok":
        logger.warning("serving with a %s backend: %s", status.status, status.detail)
    build_server().run(transport=settings.mcp_transport)


if __name__ == "__main__":
    main()
