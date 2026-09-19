# Reply to PR #1 — Buse's decisions

(Paste this as a comment on PR #1, or into the review thread — GitHub write
access isn't available from here, so this is drafted for you to post.)

All nine items below: **approved and adopted**. My local track's contracts now
match yours exactly (`retrieval_J.py`, `mcp_tools_J.py`, `eval_dataset_J.py`,
`judge_J.py`, merged `config_J.py`) — DRAFT markers can come off.

- [x] `retrieval_J.py` as drafted — **and** `year` is always populated at
      ingest (`ingestion/chunk_B.py`, from the corpus manifest).
- [x] `eval_dataset_J.py` as drafted — **and** the `Qrel.labeler="buse"`
      default is removed; `labeler` is a required field now.
- [x] `ToolError.remediation` and `ToolError.partial`.
- [x] `inferential_leap` as a sixth violation kind.
- [x] Shallow/deep violation split and its import guard.
- [x] `FaithfulnessVerdict.confidence`.
- [x] `AnswerVerdict`.
- [x] `EvidenceConflict` / `EvidenceAssessment`.
- [x] The embedding model: `config_J.Settings.embedding_model` (merged this
      week) now defaults from `configs/ingestion_B.yaml`'s real value
      (`BAAI/bge-small-en-v1.5`) via a `default_factory`, so it can't silently
      drift from what ingestion actually used. Same fix for `corpus_version`,
      sourced from `retrieval_B.yaml`'s `vector_store.collection`.

## What's landed on my side since the contracts were pinned

- `retrieval/service_B.RetrievalService.retrieve(RetrievalRequest) ->
  RetrievalResponse` — the exact name `TrackARetrieval` binds to. Stamps
  `corpus_version`/`embedding_model`/`reranker_version` on every response;
  populates `bm25_score`/`vector_score`/`rerank_score` separately.
- `ingestion/chunk_B.py` — `Chunk.derive_id` (sha256) replaces the old sha1
  scheme; nested `ChunkMetadata`; exact char-offset tracking through
  packing/overlap (verified, not approximated); correct per-chunk `page`
  attribution for sections spanning multiple pages (was defaulting to the
  section's first page for every chunk in it — fixed).
- `reranker/baseline_B.BaselineRanker` and `reranker/registry_B.CrossEncoderRanker`
  now both satisfy `contracts.retrieval_J.Reranker` (`.version` + `.rerank(...)`),
  so there's a stamped `reranker_version="baseline"` comparison point before the
  DPO model exists.
- `reranker/pairs_B.py` rewritten to consume `RelevanceVerdict` (0-3) and
  produce your `PreferencePair` directly, so the leakage test and my trainer
  read the same shape.
- `configs/reranker_B.yaml`'s `pairs.min_score_gap` retuned for the 0-3 scale
  (was carried over from an old 1-5 draft's tuning, where "2" meant something
  different) — see `scripts/sweep_min_score_gap_B.py` for the pair-yield
  analysis behind the new value, and the file's own comment for the caveats.
- `eval/thresholds_B.yaml` — added proposed per-metric `min_effect` (item #10
  from the design review), alongside the existing single `--min-effect` flag
  your gate reads today. Wire it in whenever's convenient; not blocking.
- Two bugs found and fixed along the way, in case they matter for your side
  too: (1) the ingestion pipeline built the vector store but never built/saved
  the BM25 index, so the sparse arm could never actually populate from a plain
  `make ingest`; (2) re-ingesting into the same Chroma collection left old
  chunk_ids behind instead of replacing them, silently doubling the index.

## Still open on my side (not blocking your contracts, listed for visibility)

- Real relevance labels — `llm_draft_grades_B.json`'s existing grades are
  keyed to the old chunk_id scheme and need a genuine re-read against the new
  ids, not a mechanical remap.
- `scripts/label_eval_B.py` now exists (promotes reviewed drafts into the
  frozen `queries_B.jsonl`/`qrels_B.jsonl`, refuses to silently change an
  existing grade) — but the frozen files stay empty until the labeling above
  happens.
- Eval-set content: unanswerable queries (4 candidates drafted and BM25-
  verified, see `docs/eval_set_gaps_proposal_B.md`, need ~15-20 more),
  conflict queries (not yet — needs a closer read than I could respons­ibly
  do with a keyword check), second labeler, 200-query target.
- `observability/tracing_B.py` — can't rename it on my side yet; the real
  199-line implementation only exists as `tracing_S.py` on your branch, and my
  local copy is still an empty stub. This is a post-merge rename, not
  something I can do now.
