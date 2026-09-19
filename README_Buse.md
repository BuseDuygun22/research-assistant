# Research assistant project — Buse's track

## What this project is

A retrieval-augmented research assistant with three layers: a RAG pipeline over
a paper corpus, an MCP server exposing that pipeline as callable tools, and a
LangGraph agent graph (researcher, writer, editor) that uses those tools to
produce cited drafts. An offline DPO training loop improves the retrieval
reranker over time, tracked in MLflow, and a CI/CD gate only promotes a new
reranker, prompt, or agent version if it clears an eval threshold.

Work is split into two tracks, each with a mix of domain judgment calls,
engineering, and research/ML work — nobody is stuck doing only one kind of
task. This README covers Track A. Sude's README covers Track B in the same
detail; the summary below is enough to know what to expect from her.

## Buse's responsibilities — Track A: Retrieval, Data & Evaluation

**Domain**
1. **Corpus curation and scoping** — select and bound the paper corpus (what
   topic, roughly how many papers, what's in/out of scope), and write a short
   domain brief: what questions the assistant should actually be able to
   answer, and why this corpus is the right one to answer them from. This is
   a real judgment call, not busywork — it shapes what "good retrieval" even
   means later.

**Engineering**
2. **Ingestion and chunking pipeline** — parse the papers, chunk by
   section/paragraph, tag each chunk with metadata (title, section, page),
   embed with the agreed model
3. **Vector DB and hybrid retrieval** — stand up Qdrant or Chroma, implement
   BM25 + vector hybrid search for the top-20 candidate set (this is the
   candidate set your reranker consumes and the shape the MCP tools return —
   agree the contract with Sude early)
4. **Observability** — wire Langfuse or LangSmith tracing across the whole
   system, not just the agent graph. This is worth doing early rather than
   bolted on later, since it's what makes every other bug findable.

**Research / evaluation**
5. **Held-out eval query set** — curate the query set and hand-label
   relevance, based on the domain brief from item 1
6. **Retrieval metrics** — define and compute nDCG@5, MRR, citation
   precision; log retrieval experiments in MLflow
7. **DPO reranker training** — turn Sude's relevance-judge scores into
   preference pairs, fine-tune a cross-encoder with `trl`'s DPO trainer, and
   register the tuned model behind the same interface as the baseline ranker
   so it is a drop-in swap in the retrieve-top-k step

## Sude's responsibilities (summary — see her README for detail)

- Editorial rubric (what counts as a faithfulness violation) + writing the
  results/discussion section of the final report
- MCP server implementation and CI/CD pipeline
- Relevance/faithfulness judge, LangGraph agent graph

## Build order

1. Plain RAG pipeline with MCP tools, manually evaluated — **joint**
2. Ingestion/vector DB (Buse) in parallel with the judge (Sude); DPO
   reranker (Buse) follows, once the judge is emitting scores
3. CI/CD gate (Sude) using eval thresholds Buse defines in step 5-6
4. MCP tools finalized + observability (Buse), then agent graph (Sude)

## Integration points — read this before writing code

- **MCP tool schema**: agree the Pydantic input/output models before either
  side builds against them
- **Judge output schema**: your preference-pair builder consumes Sude's
  relevance judge output directly, so pin that Pydantic model before she
  writes the judge
- **Reranker interface**: your "retrieve top-k" step loads the ranker from a
  registry, so the tuned model swaps in for the naive similarity ranking
  without the MCP tools changing
- **Eval dataset format**: your held-out query set needs a fixed schema
  Sude's CI/CD job can consume without modification

## Shared / joint work

- End-to-end integration testing
- Final write-up and demo
