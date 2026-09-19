# Research assistant project — Sude's track

## What this project is

A retrieval-augmented research assistant with three layers: a RAG pipeline over
a paper corpus, an MCP server exposing that pipeline as callable tools, and a
LangGraph agent graph (researcher, writer, editor) that uses those tools to
produce cited drafts. An offline DPO training loop improves the retrieval
reranker over time, tracked in MLflow, and a CI/CD gate only promotes a new
reranker, prompt, or agent version if it clears an eval threshold.

Work is split into two tracks, each with a mix of domain judgment calls,
engineering, and research/ML work — nobody is stuck doing only one kind of
task. This README covers Track B. Buse's README covers Track A in the same
detail; the summary below is enough to know what to expect from her.

## Sude's responsibilities — Track B: Agents, Training & Generation

**Domain**
1. **Editorial rubric** — define what counts as a faithfulness violation and
   what a "good" draft looks like for this domain, so the editor agent has
   real criteria rather than a vague "check for flaws." Also owns writing the
   results/discussion section of the final report, since that's where the
   domain framing actually gets used.

**Engineering**
2. **MCP server implementation** — FastAPI-backed tools (`search_papers`,
   `get_citation`, `summarize_section`), matching the schema agreed with Buse,
   containerized with Docker
3. **CI/CD pipeline** — GitHub Actions workflow that runs the eval suite
   (query set and thresholds from Buse) on every change, gates merge/deploy,
   automates deployment on a pass

**Research / ML**
4. **Relevance and faithfulness judge** — the LLM/RAGAS-based judge, first
   scoring retrieved chunks (feeds DPO training), later reused to check
   drafts against sources (feeds the editor agent) — one judge, two uses
5. **DPO reranker training** — fine-tune a cross-encoder with `trl`'s DPO
   trainer on preference pairs from the judge
6. **LangGraph agent graph** — state schema, researcher/writer/editor nodes,
   the conditional edge that routes back to the researcher on a flaw (capped
   at 3 revisions, falling through to a `flag_for_human` node)

## Buse's responsibilities (summary — see her README for detail)

- Corpus curation and domain scoping
- Ingestion/chunking pipeline and vector DB + hybrid retrieval
- Observability, held-out eval query set, retrieval metrics

## Build order

1. Plain RAG pipeline with MCP tools, manually evaluated — **joint**
2. DPO reranker + judge (Sude) in parallel with ingestion/vector DB (Buse)
3. CI/CD gate (Sude) using eval thresholds Buse defines in her track
4. MCP tools finalized + observability (Buse), then agent graph (Sude)

## Integration points — read this before writing code

- **MCP tool schema**: agree the Pydantic input/output models before either
  side builds against them
- **Reranker interface**: your tuned reranker needs to drop into Buse's
  "retrieve top-k" step as a swap-in replacement, not a rewrite
- **Eval dataset format**: consume Buse's held-out query set in the fixed
  schema she defines, without modifying it locally

## Shared / joint work

- End-to-end integration testing
- Final write-up and demo
