# Design review — every stage, every principle, every edge case

Status: full audit, 16 September 2026, **third revision** — after Track S was
completed and its remaining partials closed. Assesses `docs/architecture_J.md` and
the code as built.

Scope: Sude's track is **implemented** and reviewed below. Buse's track is
**audited but not modified** — Part 7 is written for her: what the code on the
other side of the seam now expects, what in her design holds up, and what to
change.

Companions: `docs/architecture_J.md` (rationale), `docs/contracts_J.md` (the
seam), `docs/editorial_rubric_S.md` (what `passed` means),
`docs/research_notes_S.md` (2026 literature sweep).

**Grades:** **Valid** designed, built, tested · **Fixed** was defective or
missing, now built and tested · **Partial** typed but incomplete · **Blocked**
needs Buse's data or code.

**State of Track S: complete, no partials.** 186 tests pass, ruff clean. The only
Sude files still empty are `docs/report/*`, which need real results.

---

## Part 1 — The architecture as built

```
   ┌──────────────────────────── ONLINE (per query) ─────────────────────────────┐
   │                                                                             │
   │   question                                                                  │
   │      │                                                                      │
   │      ▼                                                                      │
   │ ┌──────────┐ evidence ┌────────┐  write   ┌────────┐  draft  ┌────────┐      │
   │ │researcher│─────────►│ triage │─────────►│ writer │────────►│ editor │      │
   │ └──────────┘          └────────┘          └────────┘         └────────┘      │
   │   ▲  ▲  ▲               │    │  conflicts ─┘  ▲                  │           │
   │   │  │  └─ insufficient ┘    │  travel as     │                  │           │
   │   │  │     (retry, max 2)    │  instructions  └── rewrite ───────┤ shallow   │
   │   │  └────────────── re_retrieve ────────────────────────────────┤ deep      │
   │   │                          │                                   │           │
   │   │                          ▼ insufficient, no budget           ▼           │
   │   │                       abstain                  accept · abstain · escalate│
   └─────────────────────────────────────────────────────────────────────────────┘
          │
          └─► eval gate: attribution · utility · process · significance → promote/reject/REFUSE
```

- **Triage runs before the writer** (E40): a retrieved set that cannot answer the
  question is retried or abstained on without spending a writer call, and
  conflicts between passages are passed to the writer as instructions (E38).
- **Budgets:** `max_rewrites=3`, `max_re_retrievals=2`, `max_steps=12`, plus a
  **cost ledger** (rewrite 1, re-retrieve 4, human 50 units) with an opt-in cost
  ceiling.
- **Routing** is two pure functions — `decide_evidence_route` and `decide_route` —
  over verdicts and budget. No model call picks an edge.

---

## Part 2 — Stage by stage

| Stage | Owner | State | Finding |
|---|---|---|---|
| **0** Ingestion | Buse | Design valid, unbuilt | See Part 7 |
| **1–2** Retrieval | Buse | Design valid, unbuilt | See Part 7 |
| **3** MCP tools | Sude | **Complete** | Server, remediation/partial on every error, tool deadlines |
| **4** Agent graph | Sude | **Complete** | Researcher → **triage** → writer → editor; dedup of near-identical passages; novelty-checked reformulation; handover with a next step per trigger |
| **5** Judge | Sude | **Complete** | Rubric-derived `passed`, answer relevance, **evidence triage with conflict detection**, 3-judge panel, **calibration instruments** |
| **6–7** DPO | Buse | Pairs ready on Sude's side | `build_preference_pairs`; leakage test guards the split |
| **5** CI gate | Sude | **Complete** | Five refusals, paired bootstrap, Holm on p-values, effect floor, sensitivity + diagnostics (granularity, cost) |
| Observability | Sude | **Complete** | Every decision carries trigger, reason, budget and cost |

---

## Part 3 — The 40 principles

**Final: 37 Valid/Fixed · 0 Partial · 3 Blocked on Buse** (22–24, one shared cause: no labels yet).

The three partials from the previous revision, and what closed them:

| # | Principle | Was | Now |
|---|---|---|---|
| 9 | Calibrated confidence | Field existed, no procedure | **Fixed.** `judge/calibration_S.py`: temperature scaling, reliability bins with direction, expected calibration error, and `suggest_escalation_threshold` — which returns `None`, not 1.0, when no threshold reaches the target. Fitting needs Buse's labels; the instrument is tested on synthetic data so the first real run is a result, not a debugging session |
| 39 | Optimise for cost | Budgets implied cost, nothing measured it | **Fixed.** `ROUTE_COST` / `HUMAN_COST` ledger on `Budget`, `cost_spent` in every decision and in gate diagnostics, and an opt-in `cost_ceiling` escalation when finishing the budget would cost more than a person |
| 40 | Adversarial tests | Router tests only; no adversarial corpus | **Fixed.** `tests/fixtures/adversarial_corpus_S.py`: superseded result, genuine disagreement, near-duplicate, topical-but-useless, prompt injection — each run through the full graph |

Full table:

| # | Principle | Verdict | Where |
|---|---|---|---|
| 1 | Structured shared state | Fixed | `AgentState`, frozen |
| 2 | Separate responsibilities | Valid | 5 nodes; writer has no retrieval tool |
| 3 | Verifier diagnoses errors | Valid | Typed kinds + explanation |
| 4 | Explicit error taxonomy | Valid | Exhaustive and disjoint by construction |
| 5 | Route deterministically | Fixed | Two pure routing functions |
| 6 | Shallow → writing | Fixed | |
| 7 | Deep → retrieval | Fixed | |
| 8 | Never ignore unknown errors | Fixed | Import guard + runtime escalation |
| 9 | Calibrated confidence | **Fixed** | `calibration_S.py` + panel agreement |
| 10 | Confidence for escalation | Valid | Threshold derived by `suggest_escalation_threshold` |
| 11 | Verdict ≠ confidence | Valid | |
| 12 | Separate revision counters | Fixed | |
| 13 | Execution budgets | Fixed | Counters + cost ledger |
| 14 | Don't assume revision helps | Valid | Depth split |
| 15 | Re-retrieve on bad evidence | Fixed | Now also *before* writing (triage) |
| 16 | Evidence provenance | Fixed | `ClaimSpan` |
| 17 | Claim-level verification | Fixed | Plus a granularity instrument (E39) |
| 18 | Retrieval ≠ generation | Valid | |
| 19 | Strategic escalation | Fixed | 9 terminal triggers, each with a next step, test-enforced |
| 20 | Explicit termination | Valid | Tested adversarially |
| 21 | Evaluate decisions | Fixed | Trajectory metrics in the gate |
| 22 | Calibrate against humans | **Blocked** | Instrument ready; needs qrels |
| 23 | Inter-rater agreement | **Blocked** | `krippendorff_alpha_nominal` ready; needs a second labeller |
| 24 | Inspect disagreements | **Blocked** | `AgreementReport` returns the matrix + worst cells; needs labels |
| 25–27 | Paired, bootstrap, CIs | Fixed | |
| 28 | Multiple comparisons | Fixed | Holm on bootstrap p-values |
| 29 | Effect thresholds | Fixed | Lower CI bound ≥ `min_effect` |
| 30 | Trace the workflow | Fixed | |
| 31 | Observable routing | Fixed | |
| 32 | No LLM-controlled loops | Valid | Includes the prompt-injection test |
| 33 | Design for failure | Fixed | Tool timeouts; triage failure degrades rather than halts |
| 34 | Validate outputs | Valid | |
| 35 | Prevent state corruption | Fixed | |
| 36 | Targeted correction | Fixed | |
| 37 | Preserve prior work | Fixed | |
| 38 | Evidence vs writing failure | Valid | |
| 39 | Cost as well as quality | **Fixed** | Cost ledger |
| 40 | Adversarial testing | **Fixed** | Adversarial corpus |

### Defects found in this track's own earlier work

1. **Fail-open routing default** — unclassified violation kinds routed to the
   writer. Fixed with import-time and runtime guards.
2. **A Holm correction that corrected nothing** — it compared the fixed 0.95
   confidence level against the corrected threshold, rejecting every metric
   whenever more than one was tested. Now runs on bootstrap p-values.
3. **Near-duplicates counted as corroboration** — a preprint and its published
   version entered the evidence as two agreeing sources. Now deduplicated by
   normalised text on merge. Found by the adversarial corpus, which is the case
   for having one.

---

## Part 4 — Edge cases

| | Case | Behaviour now | |
|---|---|---|---|
| E1 | Unclassified violation kind | Import fails; runtime escalates | ✅ |
| E2 | `passed=False`, no violations | Escalate | ✅ |
| E3 | Deep violation, no retrieval budget | Escalate, never to writer | ✅ |
| E4 | Same deep violation repeats | Re-retrieve to budget, escalate | ✅ |
| E5 | Passed, low confidence | Escalate first | ✅ |
| **E6** | **Faithfully cites a wrong source** | **Undetected** | ❌ **limit of the approach** — needs reference answers |
| E7 | One-sided evidence | Conflicts caught when both sides are retrieved (E38); `context_recall` catches the rest in eval | ⚠️ runtime needs both sides present |
| E8 | Quotes only, answers nothing | `re_retrieve` (`answer_incomplete`) | ✅ |
| E9 | Corpus cannot answer | Abstain — now at triage, before a writer call | ✅ |
| E10 | Answerable, retrieval missed | Re-retrieve | ✅ |
| E11 | "A. B. Therefore C." | `inferential_leap` if seen; granularity instrument measures the risk | ⚠️ measurable, not solved |
| E12 | Repeated reformulation | Novelty check | ✅ |
| E13–E15 | Zero results / budget exhausted / zero budget | Named triggers | ✅ |
| E16–E17 | Malformed judge output | Editor escalates; triage degrades | ✅ |
| E18 | Judge confidently wrong | Panel disagreement lowers confidence; reliability bins expose overconfidence | ⚠️ needs labels to confirm |
| E19 | Zero claims | Precision 0.0, not 1.0 | ✅ |
| E20 | Grade-0-dominated qrels | `AgreementReport.is_degenerate` flags it | ⚠️ **[Buse]** fix is in sampling |
| E21 | Panel splits | Confidence 0 → escalate | ✅ |
| E22–E30 | Gate cases | Refusals, Holm, effect floor, seeded | ✅ |
| E31–E37 | State and tools | Ownership, partial, timeout, stub, stale ids, frozen state | ✅ |
| **E38** | **Two sources contradict each other** | Triage classifies freshness / disagreement / scope; writer told to report, not resolve | ✅ **fixed** |
| **E39** | **Over-fine decomposition** | `claim_granularity` in gate diagnostics; `granularity_sweep` picks a level | ✅ **instrumented** |
| **E40** | **Bad evidence discovered after writing** | Triage before the writer | ✅ **fixed** |
| E41 | Preprint + published duplicate | Deduplicated on merge | ✅ **fixed** |
| E42 | Passage contains a prompt injection | Routing is code; a fooled judge changes a verdict, never the rules | ✅ tested |
| E43 | Triage judge fails | Degrades to full draft-and-judge path | ✅ |
| E44 | Conflict names a passage never retrieved | Dropped | ✅ |
| E45 | Judge says `sufficient` but lists conflicts | Specific finding wins → `conflicting` | ✅ |

---

## Part 5 — Comparable systems, and where we now stand

| System | Their edge case | Ours now |
|---|---|---|
| **CRAG** (2401.15884) | Irrelevant retrieval → fluent wrong answer; evaluator before generation, three labels | **Adopted.** Triage before the writer, four labels (`partial` added) |
| **ALCE** + *Are Finer Citations Always Better?* | Citation failure taxonomy; attribution peaks at intermediate granularity | Taxonomy maps cleanly. Granularity is now **measured** rather than assumed atomic |
| **MAST** (2503.13657) | Incorrect verification; premature termination; κ = 0.88 annotation | Named-trigger termination, rubric verification, calibration instruments. κ awaits labels |
| **AbstentionBench / RefusalBench** | Sub-50% refusal accuracy; prompt-based abstention fails | Abstention is a routed edge, now decided at triage too |
| **ConflictRAG / DRAGged into Conflicts** | RAG assumes consistent evidence; freshness vs disagreement | **Adopted.** Conflicts detected, classified, and carried to the writer as instructions |

---

## Part 6 — Sude: remaining work

No design gaps remain on Track S. What is left needs data:

| Action | Needs |
|---|---|
| Fit `TemperatureScaler`, derive `escalation_confidence` | Qrels + judge verdicts on them |
| Run `granularity_sweep` (atomic vs clause vs sentence) | A real corpus and the judge |
| Replace `ROUTE_COST` estimates with measured span timings | Runs against the real backend |
| Write `docs/report/*` | Results |

---

## Part 7 — Buse's track: audit and suggestions

Written for Buse. Nothing in her files was modified.

### 7.1 What the code on the other side of the seam now expects

These are hard requirements — Track S binds to them lazily today and will use them
the moment they exist.

| File | Must provide | Why it matters on Sude's side |
|---|---|---|
| `retrieval/service_B.py` | `class RetrievalService` with `retrieve(RetrievalRequest) -> RetrievalResponse` and `get_chunk(chunk_id) -> Chunk \| None` | `TrackARetrieval` imports exactly these names. Until they exist every run uses the stub and the gate refuses to promote |
| same | Honour `candidate_k`, `filters` (`year_min`/`year_max`), `use_reranker` | The tool layer passes all three; ignoring filters makes the "drop the year filter" remediation meaningless |
| same | Fill `bm25_score`, `vector_score`, `rerank_score` **separately**; 1-based `rank`; stamp `corpus_version`, `embedding_model`, `reranker_version` | The gate refuses to compare runs whose stamps differ; blank stamps compare as equal and defeat that check |
| `eval/metrics/retrieval_B.py` | `ndcg_at_k(retrieved: list[str], relevant: dict[str, int], k: int) -> float` and `mrr(retrieved, relevant) -> float` | The gate looks for these names. Without them it gates on generation metrics only and logs that retrieval is *not* gated |
| same | Canonical `context_recall` | A provisional copy lives in `generation_S.py`; it should defer to hers so the two cannot drift |
| `eval/datasets/queries_B.jsonl` | `EvalQuery` rows: `query_id, query, intent, split, notes` | Strict loader: one malformed row fails the gate by design |
| `eval/datasets/qrels_B.jsonl` | `Qrel` rows: `query_id, chunk_id, grade (0-3), labeler` | `chunk_id` must come from `Chunk.derive_id`, or every label silently rots on re-ingest |
| `reranker/*_B.py` | Implement the `Reranker` protocol: `version`, `rerank(query, candidates, top_k)` setting `rerank_score` and `rank` | A no-op reranker is a valid baseline and should exist first, so the baseline run has a stamp |
| `data/preference_pairs.jsonl` | Train only on pairs from `build_preference_pairs`, filtered on `margin` | `test_no_eval_leakage_J.py` fails if any pair's query is in the test split |

### 7.2 What in her design holds up

| Decision | Verdict |
|---|---|
| Structural chunks, not fixed windows | **Valid** — citations point at a checkable span |
| Deterministic `chunk_id` hash | **Valid** — the strongest rule in the repo |
| Hybrid BM25 + dense | **Valid** — complementary blind spots, well evidenced |
| RRF over weighted blending | **Valid** — no normalisation constant to rot. The adversarial corpus uses exactly this as its superseded-result example |
| ~4× `top_k` candidate pool | **Valid** — inside the published 50–200 band |
| Separate score columns | **Valid** — makes a metric move attributable |
| DPO over reward model + PPO | **Valid** for a two-person project |
| Margin retained on pairs | **Valid** — lets her tune without regenerating |

### 7.3 Suggestions, in priority order

1. **Double-label a 50-item qrels subset** with a second annotator (principle 23).
   One annotator cannot tell you whether the labels the judge is calibrated
   against are reliable. `krippendorff_alpha_nominal` and `AgreementReport` are
   ready to score it. Precedent: MAST reports κ = 0.88 as part of its result.
   *Related joint-contract suggestion:* drop the `labeler="buse"` default so the
   annotator is always explicit.

2. **Size the eval set at ~200 queries.** That resolves a 4% nDCG change at a
   typical per-query spread of 0.20; a 3% change at spread 0.25 would need ~550.
   Measure the real spread on a 20–30 query pilot first — `observed_sd` computes
   it — then pin the number.

3. **Include `intent="unanswerable"` queries** — roughly 10–15% of the set.
   Without them abstention recall is literally uncomputable, and abstention is a
   headline claim of the design.

4. **Include conflict queries.** Questions whose relevant chunks include a
   superseded result or a genuine disagreement. E38 detection now exists on Sude's
   side, but it can only be *evaluated* if the eval set contains cases that need it.

5. **Stratify qrels so grade 0 does not dominate** (E20). If more than ~80% of
   labels are 0, kappa becomes unstable and a judge that always answers 0 looks
   acceptable — `AgreementReport.is_degenerate` will flag it, but the fix is in
   how candidates are sampled for labelling.

6. **Always populate `ChunkMetadata.year`**, plus `section`, `page`, and
   `char_start`/`char_end`. Freshness conflict detection reads `year`; citations
   read the rest. A chunk without a year cannot be recognised as superseded.

7. **Record parse coverage per document and refuse to index below a threshold.**
   A PDF whose middle pages are scanned images produces a silent hole that reads
   downstream as a retrieval failure.

8. **Deduplicate at ingest, not only at query time.** Sude's side now drops
   exact-text duplicates when merging evidence, but that only catches identical
   text. A preprint and its revised publication differ slightly. Linking versions
   of the same paper at ingest (a shared canonical `paper_id`) is the real fix.

9. **Ship a no-op `Reranker` first**, stamped `reranker_version="baseline"`, so
   the first real gate run has a baseline to compare the DPO reranker against.

10. **Put per-metric minimum effects in `eval/thresholds_B.yaml`**, e.g.
    `min_effect: {ndcg@5: 0.02, mrr: 0.02}`. The gate currently takes one
    `--min-effect` flag; a per-metric file is the agreed place for the number she
    owns, and Sude will wire the gate to read it once the format is settled.

11. **Answer the open questions in `architecture_J.md` §5** — embedding model,
    manual `corpus_version` bumping, reranker ownership, tracing ownership. The
    embedding model matters most: until it is fixed, `embedding_model` is free
    text and the stamp check compares strings.

12. **Reference answers for a subset** (joint with Sude). The only way to detect
    E6, a faithfully cited wrong source. Even 30–50 queries would let the report
    state correctness on a measured subsample instead of leaving it unmeasured.

### 7.4 Buse's track at a glance

| | Items |
|---|---|
| **Blocks Sude now** | `RetrievalService`, `ndcg_at_k`/`mrr`, eval dataset |
| **Blocks calibration** | Qrels, and a second labeller on a subset |
| **Improves correctness** | Year metadata, parse coverage, ingest-time dedup, stratified qrels |
| **Blocks the headline claims** | Unanswerable queries (abstention), conflict queries (E38), reference answers (E6) |

---

**Summary.** Track S is complete with no partials: triage before writing, conflict
detection carried to the writer, calibration and agreement instruments, a cost
ledger, and an adversarial corpus that already caught one real defect. Every
remaining principle gap has the same cause: labels and a corpus that do not exist
yet. Part 7 lists, for Buse, the interface the code expects and the twelve changes
that would let the system's headline claims be measured rather than asserted.
