# Integration contracts — JOINT

Status: **DRAFT, awaiting Buse's sign-off.** Proposed by Sude.
Contract version: `0.1.0-draft` (`src/research_assistant/contracts/__init__.py`).

Both READMEs say the same thing: agree these before either side builds against
them. This document is the agreement; the Pydantic models in
`src/research_assistant/contracts/` are the enforcement. Where the two disagree,
the code wins and this document is stale — fix it in the same PR.

## Rule of the seam

Neither track imports the other's internals. Track A owns ingestion, the vector
store and hybrid search; Track B owns the judge, MCP tools and the agent graph.
They meet at exactly four places, listed below. Anything that needs to cross the
seam and is not here needs a PR to this file first.

---

## 1. `retrieval_J.py` — chunks and retrieval

Owner of the data: **Buse**. Consumer: **Sude**.

- `Chunk` — `chunk_id`, `text`, `metadata`. `ChunkMetadata` carries everything a
  citation needs: `paper_id`, `title`, `authors`, `year`, `section`, `page`,
  `char_start/char_end`, `source_uri`.
- **`chunk_id` must be minted by `Chunk.derive_id(paper_id, char_start, char_end, text)`.**
  This is the load-bearing rule of the whole project. The eval qrels label
  `chunk_id`s by hand; if a re-ingest reshuffles ids, every label silently rots
  and the CI gate starts measuring noise. Deterministic ids mean a re-ingest of
  the same corpus either reproduces the ids or visibly breaks.
- `RetrievedChunk` keeps `bm25_score`, `vector_score` and `rerank_score` as
  separate fields alongside the effective `score`. Collapsing them into one
  number would make it impossible to attribute a metric change to the lexical
  leg, the dense leg, or the reranker.
- `RetrievalResponse` stamps `corpus_version`, `embedding_model` and
  `reranker_version`. The eval gate refuses to compare two runs whose stamps
  differ — otherwise a "+4 nDCG" that was really a corpus change gets promoted.

### The reranker seam

```python
class Reranker(Protocol):
    version: str
    def rerank(self, query, candidates, top_k) -> list[RetrievedChunk]: ...
```

Buse's `retrieve()` takes an optional `Reranker` and calls it on the
`candidate_k` pool. A no-op reranker returning candidates unchanged is a valid
implementation, which is what keeps the baseline path exercised by the same code
path as the tuned one. **Ownership note:** the DPO-tuned cross-encoder is
implemented by Buse (`reranker/*_B.py`), against preference pairs Sude's judge
emits. Sude never imports the reranker; she only produces `PreferencePair` rows.

---

## 2. `mcp_tools_J.py` — the three MCP tools

Owner: **Sude**. Consumer: both, plus the agent graph.

| Tool | Input | Output |
|---|---|---|
| `search_papers` | `query`, `top_k`, `year_min/max`, `use_reranker` | `results: list[RetrievedChunk]`, `corpus_version`, `reranker_version` |
| `get_citation` | `chunk_id`, `style` (`apa`/`bibtex`/`inline`) | `formatted`, `quote`, `paper_id`, `title`, `page` |
| `summarize_section` | `chunk_ids`, `focus`, `max_words` | `summary`, `grounded_in` |

Two rules:

- **Tools never raise across the MCP boundary.** Failures come back as a
  `ToolError` with a `code` and a `retryable` flag. An agent that receives a
  stack trace cannot recover; one that receives `retryable=True` can.

  `ToolError` also carries `remediation` and `partial`. `retryable` says a retry
  is *permitted*; `remediation` says which retry is *worth making* — the
  difference between an agent that reformulates and one that reissues the same
  failing call three times. `partial=True` marks a call that returned usable
  results alongside the error, so `summarize_section` can return four of five
  chunks instead of failing whole. Both are optional and default to the previous
  behaviour, so neither breaks an existing caller.
- **`get_citation` returns a verbatim `quote` and its character span.** This is
  what makes a citation checkable rather than merely plausible — the
  faithfulness judge verifies each claim against that span, not against the
  whole paper.

---

## 3. `judge_J.py` — verdicts and preference pairs

Owner: **Sude**. Consumers: the DPO trainer (Buse) and the editor agent (Sude).

- `RelevanceVerdict.grade` is graded `0..3` on the TREC convention (0 irrelevant,
  1 marginal, 2 relevant, 3 directly answers) — the same scale as Buse's qrels.
  Binary relevance would make nDCG degenerate and would leave DPO pairs with no
  margin to filter on.
- `FaithfulnessVerdict` carries typed `violations`, `citation_precision`,
  `coverage` and a derived `passed`. `passed` is computed by the rubric
  (`docs/editorial_rubric_S.md`), not self-reported by the model, because the
  editor agent routes on it.
- **Violations are split by depth, and the editor routes on the split.**
  `DEEP_VIOLATION_KINDS` marks the violations a rewrite cannot repair
  (`unsupported_claim`, `inferential_leap`, `contradicts_source`) because the
  evidence needed to repair them is not in the writer's context. Those go back to
  the researcher for re-retrieval; the shallow kinds (`missing_citation`,
  `overstated_certainty`, `misattributed_citation`) go back to the writer. A
  writer told to fix an unsupported claim without new evidence can only delete it
  or invent support, and only one of those is behaviour we want.
  `FaithfulnessVerdict.revision_route()` is that rule, derived from the verdict
  alone so both tracks read one implementation. Rationale and sources:
  `docs/research_notes_S.md` §2.1.
- `inferential_leap` is deliberately distinct from `unsupported_claim`: the
  source supports A and the draft asserts B, a plausible but unsupported step.
  2026 evaluations of deep-research agents attribute most residual citation error
  to that pattern rather than to wholly uncited claims, so it earns its own label
  and its own rubric language.
- `FaithfulnessVerdict.confidence` is the *calibrated* probability that `passed`
  is correct — not the model's self-report. It exists so `flag_for_human` can be
  an uncertainty signal rather than only a budget timeout: a confident pass and
  an unsure pass should not take the same edge. It defaults to `1.0`, so an
  uncalibrated judge degrades to the previous always-trust behaviour instead of
  escalating everything. The threshold lives in `Settings.escalation_confidence`
  and ships at `0.0` — escalate on nothing — until the judge is calibrated
  against Buse's qrels and we know what a given number means.
- `PreferencePair` is the handoff to Buse's trainer: `(query, chosen, rejected,
  margin)` as JSONL. `margin` is retained so training can drop near-ties —
  judge-built pairs with a thin grade gap are the main source of label noise in
  offline DPO, and dropping them is cheaper than training through them.
- Every verdict carries `JudgeMeta` (`judge_model`, `prompt_version`,
  `temperature`). Without it a metric shift cannot be attributed to the system
  versus a judge-prompt edit.

---

## 4. `eval_dataset_J.py` — the held-out set

Owner: **Buse**. Consumer: **Sude's CI gate**.

- `eval/datasets/queries_B.jsonl` → `EvalQuery(query_id, query, intent, split, notes)`
- `eval/datasets/qrels_B.jsonl` → `Qrel(query_id, chunk_id, grade, labeler)`
- `load_eval_dataset()` is **strict**: a malformed row fails the gate rather than
  silently shrinking the eval set, which would quietly make the gate easier to
  pass.
- Sude's CI job reads these files and never writes them. Threshold values live in
  `eval/thresholds_B.yaml`, also Buse's.
- The `test` split is never used for prompt or reranker selection; `dev` is.
  `tests/test_no_eval_leakage_J.py` enforces that mechanically.

---

## Open questions for Buse

1. **Reranker files are suffixed `_B` but Sude's README assigns DPO training to
   her.** Resolved for now as: Buse owns the training, Sude supplies the judge
   and preference pairs. Confirm.
2. **`observability/tracing_S.py` is suffixed `_S` but Buse's README assigns
   observability to her.** Sude is implementing it, since the spans live in the
   MCP/judge/agent layers she owns. Buse to add ingestion/retrieval spans — say
   if you'd rather rename it `_J`.
3. Confirm `corpus_version` bumping is manual, and confirm the embedding model
   so `RetrievalResponse.embedding_model` is not free text.
4. **How many eval queries are you labelling, and does that number let the gate
   see the effect we care about?** This one blocks you rather than the other way
   round, so it is worth answering before the qrels are finished rather than
   after. The gate now compares baseline and candidate with a *paired* bootstrap
   (`eval/metrics/significance_S.py`), which cancels per-query difficulty and so
   detects a much smaller true effect from the same labels. What it needs from
   you is the query count. Running `required_queries(effect, sd)` gives:

   | Effect to detect | sd of per-query diff = 0.15 | 0.20 | 0.25 |
   |---|---|---|---|
   | 3% | 197 | 349 | 546 |
   | 4% | 111 | 197 | 307 |
   | 5% | 71 | 126 | 197 |

   The `sd` column is the standard deviation of the *difference* between two runs
   on the same query, not of either run's scores — measurable from a pilot of
   20–30 queries in an afternoon, and currently the largest guess in the table.
   Rough read: **~200 queries** is the sensible target if we want to resolve a 4%
   nDCG change at typical spread, and 5% is comfortably in reach; a 3% change at
   the high-variance end would need closer to 550, which is probably not worth
   the labelling time. Tell us the pilot `sd` and we can pin this properly.
5. The three changes above (`ToolError.remediation`/`partial`,
   `inferential_leap` + depth routing, `FaithfulnessVerdict.confidence`) are
   implemented and tested, but they are contract edits and therefore yours to
   veto. All three are additive with backward-compatible defaults, so nothing you
   have built against these models breaks either way.
