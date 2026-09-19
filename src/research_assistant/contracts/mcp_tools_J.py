"""MCP tool I/O schemas (JOINT — Sude implements, both call).

One model per tool, named after the tool. These are what the MCP server
advertises and what the agent nodes bind to, so a change here is a breaking
change for both tracks.

DRAFT — requires Buse's sign-off before either side builds against it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .retrieval_J import Chunk, RetrievedChunk


class ToolError(BaseModel):
    """Tools never raise across the MCP boundary; they return this.

    An agent that receives a stack trace cannot recover. An agent that receives
    `retryable=True` and a reason can — and an agent that receives `remediation`
    does not have to guess *which* retry is worth making.
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="e.g. 'empty_corpus', 'chunk_not_found'.")
    message: str
    retryable: bool = False
    remediation: str | None = Field(
        None,
        description="Concrete next action for the agent, e.g. 'broaden the query or "
        "drop the year filter'. `retryable` says a retry is permitted; this says "
        "which retry is worth making. Omit rather than pad — an unhelpful "
        "remediation is worse than none, because the agent will follow it.",
    )
    partial: bool = Field(
        False,
        description="True when the call also returned usable results alongside this "
        "error — four of five chunks summarised, say. Lets a tool degrade instead "
        "of failing whole, which is the difference between a recoverable gap and a "
        "dead branch of the graph.",
    )


# --- search_papers -----------------------------------------------------------


class SearchPapersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Natural-language research question.")
    top_k: int = Field(5, ge=1, le=20)
    year_min: int | None = None
    year_max: int | None = None
    use_reranker: bool = True


class SearchPapersOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[RetrievedChunk] = Field(default_factory=list)
    corpus_version: str
    reranker_version: str
    error: ToolError | None = None


# --- get_citation ------------------------------------------------------------


class GetCitationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., description="From a prior search_papers result.")
    style: str = Field("apa", pattern="^(apa|bibtex|inline)$")


class GetCitationOutput(BaseModel):
    """`quote` + `char_start/end` are what make a citation checkable rather than
    merely plausible — the faithfulness judge verifies the claim against this
    span, not against the whole paper."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    formatted: str = Field("", description="Rendered citation in the requested style.")
    quote: str = Field("", description="Verbatim supporting span from the chunk.")
    paper_id: str = ""
    title: str = ""
    page: int | None = None
    error: ToolError | None = None


# --- summarize_section -------------------------------------------------------


class SummarizeSectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_ids: list[str] = Field(..., min_length=1, max_length=20)
    focus: str | None = Field(None, description="Optional question to summarise toward.")
    max_words: int = Field(200, ge=50, le=800)


class SummarizeSectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    grounded_in: list[str] = Field(
        default_factory=list, description="chunk_ids actually used — subset of the input."
    )
    error: ToolError | None = None


# --- registry ----------------------------------------------------------------

TOOL_SCHEMAS: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    "search_papers": (SearchPapersInput, SearchPapersOutput),
    "get_citation": (GetCitationInput, GetCitationOutput),
    "summarize_section": (SummarizeSectionInput, SummarizeSectionOutput),
}

__all__ = [
    "Chunk",
    "ToolError",
    "SearchPapersInput",
    "SearchPapersOutput",
    "GetCitationInput",
    "GetCitationOutput",
    "SummarizeSectionInput",
    "SummarizeSectionOutput",
    "TOOL_SCHEMAS",
]
