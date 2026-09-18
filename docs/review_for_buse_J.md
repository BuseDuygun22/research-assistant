# For Buse: contracts to approve, and your track's audit

From Sude, September 2026. This is the one document to read first. It covers the
two things that need your decision before anything merges.

**1. The shared contracts (`src/research_assistant/contracts/*_J.py`).** These
files are empty in git, so everything in them is new to you — both Sude's original
draft and the additions made while building Track S. They are the seam between our
tracks: a change here changes what both sides build against, which is why they stay
marked DRAFT until you sign off.

**2. The audit of your track** — what the code on Sude's side now expects from
yours, which of your design decisions hold up, and twelve suggestions. That lives
in `docs/design_review_J.md` **Part 7**, and is not repeated here.

Nothing in your files (`*_B.py`, `*_B.yaml`, `*_B.jsonl`, `domain_brief_B.md`) was
modified.

---

## How to review

Each item below is either **[Original draft]** or **[Added]** during Track S. For
each one, the question is: *approve*, *change*, or *veto*. Every [Added] item is
additive with a backward-compatible default, so a veto does not break anything
built against the rest.

The tests that pin this behaviour are in `tests/contracts/test_contracts_J.py`. A
failing contract test means "renegotiate the seam", not "fix the test".

---

## `retrieval_J.py` — mostly yours to live with

| Item | Status | What it commits you to |
|---|---|---|
| `Chunk.derive_id(paper_id, char_start, char_end, text)` | Original draft | Chunk ids are a deterministic hash. Re-ingesting the same text reproduces the id; changing chunking visibly changes it. **Your qrels depend on this.** |
| `ChunkMetadata`: `paper_id, title, authors, year, venue, section, page, char_start, char_end, source_uri` | Original draft | Optional fields, but see below — `year` is now read by conflict detection |
| `RetrievedChunk`: separate `bm25_score`, `vector_score`, `rerank_score`, 1-based `rank` | Original draft | Lets a metric change be attributed to one retrieval leg |
| `RetrievalRequest`: `query, top_k (≤50), candidate_k (≤200), filters, use_reranker` | Original draft | Your service must honour all of them |
| `RetrievalResponse` stamps: `corpus_version, embedding_model, reranker_version` | Original draft | The eval gate refuses to compare runs whose stamps differ |
| `Reranker` protocol: `version` + `rerank(query, candidates, top_k)` | Original draft | Your DPO reranker implements this; a no-op is a valid baseline |

**Decision needed:** `year` is optional in the schema, but a chunk without a year
cannot be recognised as superseded by a newer result. *Suggest:* keep it optional in
the type, but always populate it at ingest.

## `eval_dataset_J.py` — you produce these, Sude's gate consumes them

| Item | Status | Notes |
|---|---|---|
| `EvalQuery`: `query_id, query, intent, split, notes` | Original draft | `intent ∈ {factoid, comparison, synthesis, method, unanswerable}`; `split ∈ {dev, test}` |
| `Qrel`: `query_id, chunk_id, grade 0–3, labeler` | Original draft | TREC grades, same scale the judge uses |
| `load_eval_dataset` is strict | Original draft | One malformed row fails the gate on purpose; skipping rows would shrink the set silently |

**Decision needed:** `Qrel.labeler` defaults to `"buse"`. That encodes one annotator
into the schema, and judge calibration needs a second labeller on a subset to know
whether the labels are reliable. *Suggest:* remove the default so every label names
its annotator explicitly.

## `mcp_tools_J.py` — Sude implements, both call

| Item | Status | Why |
|---|---|---|
| One input/output model per tool: `search_papers`, `get_citation`, `summarize_section` | Original draft | What the MCP server advertises and the agents bind to |
| `ToolError`: `code, message, retryable`; tools never raise across the boundary | Original draft | An agent can recover from a typed error, not from a stack trace |
| `ToolError.remediation: str \| None` | **Added** | Says *which* retry is worth making, not just that one is allowed. Default `None` |
| `ToolError.partial: bool` | **Added** | A tool can return usable partial results with an error — e.g. 4 of 5 chunks summarised — instead of failing whole. Default `False` |

Low impact on your track: you call the tools, you don't implement them.

## `judge_J.py` — Sude produces; your DPO trainer reads `PreferencePair`

| Item | Status | Why |
|---|---|---|
| `JudgeMeta` stamped on every verdict | Original draft | A metric shift can be attributed to the system or to a prompt edit |
| `RelevanceVerdict` grade 0–3 + confidence | Original draft | Directly comparable to your qrels |
| `PreferencePair` with `margin` retained | Original draft | **Your trainer's input.** Filter near-ties by margin at train time without regenerating pairs |
| `FaithfulnessVerdict`: violations, `citation_precision`, `coverage`, rubric-derived `passed` | Original draft | `passed` is computed from counted violations, never self-reported by the model |
| Violation kind `inferential_leap` | **Added** | The source supports A; the draft asserts B. Separate from `unsupported_claim` because it is the most common citation error in 2026 evaluations and needs different repair |
| `SHALLOW_VIOLATION_KINDS` / `DEEP_VIOLATION_KINDS` + import guard | **Added** | Shallow errors go back to the writer, deep ones back to retrieval. The module refuses to import if a kind is in neither set, so a new kind can't be routed to the wrong place by default |
| `FaithfulnessVerdict.confidence` | **Added** | Lets an unsure verdict go to a human. Defaults to `1.0`, so a judge that doesn't report confidence behaves exactly as before |
| `AnswerVerdict` | **Added** | Does the draft answer the *question*? Faithfulness alone rewards a draft of verbatim quotes that answers nothing |
| `EvidenceConflict`, `EvidenceAssessment` | **Added** | Judged *before* writing: whether the retrieved passages can answer the question, and whether two of them contradict each other (superseded result vs genuine disagreement) |

**One note for your trainer:** `JudgeMeta.temperature` is recorded as `0.0`, but
Claude 5 models no longer accept a temperature setting, so on a real model that
field is aspirational. Judge run-to-run variance is measured instead (see
`docs/architecture_J.md`, amended temperature decision).

---

## Decisions checklist

Reply on the pull request with a line per item, or tick them here.

- [ ] `retrieval_J.py` as drafted — and agree to always populate `year` at ingest
- [ ] `eval_dataset_J.py` as drafted — and remove the `Qrel.labeler="buse"` default
- [ ] `ToolError.remediation` and `ToolError.partial`
- [ ] `inferential_leap` as a sixth violation kind
- [ ] Shallow/deep violation split and its import guard
- [ ] `FaithfulnessVerdict.confidence`
- [ ] `AnswerVerdict`
- [ ] `EvidenceConflict` / `EvidenceAssessment`
- [ ] The embedding model, so `embedding_model` stops being free text
      (`docs/architecture_J.md` §5, open question 3)

Once these are settled, the DRAFT markers come off and both tracks build against
the same contracts.

## Then, for your own track

Read `docs/design_review_J.md` Part 7. The three things that unblock Sude soonest:

1. `RetrievalService` with `retrieve` and `get_chunk` in `retrieval/service_B.py`
2. `ndcg_at_k(retrieved, relevant, k)` and `mrr(retrieved, relevant)` in
   `eval/metrics/retrieval_B.py`
3. A first eval dataset — ideally ~200 queries including unanswerable and
   conflicting ones, with a 50-item subset labelled by a second person
