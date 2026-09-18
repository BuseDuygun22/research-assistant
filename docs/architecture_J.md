# End-to-end architecture and design rationale — JOINT

Status: proposed by Sude, awaiting Buse's review.
Companion documents: `docs/contracts_J.md` (the interfaces), `docs/domain_brief_B.md`
(what the corpus is for), `docs/editorial_rubric_S.md` (what a good draft is).

Every design choice below carries an evidence tag. Nothing is here because it is
conventional.

| Tag | Means |
|---|---|
| **[P]** | Peer-reviewed or arXiv paper |
| **[B]** | Published benchmark or measured industry report |
| **[E]** | Engineering constraint we can demonstrate ourselves (a test, a measurement) |
| **[D]** | Team decision — no external evidence, recorded so it can be challenged |

---

## 1. The pipeline in one pass

A question enters; a cited draft leaves. Seven stages:

```
                    ┌─────────────────────── OFFLINE (batch) ───────────────────────┐
                    │                                                               │
  papers ──► [0] ingest ──► [1] index ─┐                    [6] judge ──► pairs ──► [7] DPO
             parse           BM25 +    │                        ▲                    train
             chunk           vectors   │                        │                      │
             embed                     │                        │                      ▼
                                       │                   preference             reranker.v2
                    ┌──────────────────┼── ONLINE (per query) ─┼──────────────────────┐│
                    │                  ▼                        │                     ││
   question ──► [3] MCP tools ──► [2] hybrid retrieve ──► rerank ◄─────────────────────┘│
                    │             BM25 + dense              (cross-encoder)             │
                    │             fused by RRF                    │                     │
                    │                                             ▼                     │
                    └──► [4] agent graph: researcher ──► writer ──► editor ──┐          │
                                               ▲            ▲                │          │
                                               │            └─ shallow ──────┤          │
                                               └─ deep ──────────────────────┤          │
                                                                             ▼          │
                                                              cited draft / flag_for_human
                    │                                                                   │
                    └──► [5] eval gate: nDCG@5, MRR, citation precision, faithfulness ───┘
                                        promote or reject
```

Ownership at a glance:

| Stage | What | Owner | Files |
|---|---|---|---|
| 0 | Corpus scoping, parse, chunk, embed | **Buse** | `docs/domain_brief_B.md`, `ingestion/*_B.py` |
| 1 | Vector store + BM25 index | **Buse** | `retrieval/vector_store_B.py`, `bm25_B.py` |
| 2 | Hybrid retrieval + fusion + reranker hook | **Buse** | `retrieval/hybrid_B.py`, `service_B.py` |
| 3 | MCP server + FastAPI tool logic | **Sude** | `mcp_server/*_S.py`, `scripts/serve_S.py` |
| 4 | LangGraph agent graph | **Sude** | `agents/*_S.py` |
| 5 | Eval gate + CI/CD | **Sude** (thresholds & query set from Buse) | `eval/run_gate_S.py`, `.github/workflows/*_S.yml` |
| 6 | Relevance + faithfulness judge | **Sude** | `judge/*_S.py` |
| 7 | DPO reranker training | **Buse** (pairs from Sude) | `reranker/*_B.py`, `scripts/train_reranker_B.py` |
| — | Retrieval metrics, eval query set, qrels | **Buse** | `eval/metrics/retrieval_B.py`, `eval/datasets/*_B.jsonl` |
| — | Tracing | **Sude** | `observability/tracing_S.py` |
| — | Contracts | **Joint** | `contracts/*_J.py` |

The loop that makes this a research project rather than a demo: stage 6 turns
judge verdicts into preference pairs, stage 7 trains a better reranker on them,
stage 2 swaps it in, stage 5 decides whether it earned promotion.

---

## 2. Stage by stage, with the reasoning

### Stage 0 — Ingestion and chunking (Buse)

Parse PDFs, split by section/paragraph, tag with `title`, `section`, `page`,
`char_start/char_end`, embed, write to the store.

**Decision: chunk on section/paragraph boundaries, not fixed token windows.** [D][E]
Citations must point at something a human can check. A fixed 512-token window
cuts mid-argument, so a "supporting span" often contains half a claim. Structural
chunks keep `section` and `page` meaningful, which is what `get_citation` returns.
Cost: variable chunk lengths, which BM25 length normalisation already handles.

**Decision: `chunk_id` is a deterministic hash of `(paper_id, char_start, char_end, text)`.** [E]
This is the single most load-bearing rule in the repo. Buse hand-labels qrels
against `chunk_id`s. If a re-ingest reshuffles ids, every label silently rots and
the CI gate starts measuring noise while still going green. With a derived id, a
re-ingest of the same text reproduces the id exactly, and a changed chunking
strategy produces *visibly* different ids — a loud failure instead of a quiet one.
Enforced in `Chunk.derive_id`.

### Stage 1–2 — Hybrid retrieval and fusion (Buse)

Query BM25 and the dense index in parallel, fuse, hand the pool to the reranker.

**Decision: hybrid, not dense-only.** [B] Sparse and dense retrieval have
systematic and *different* blind spots — BM25 misses paraphrase, dense misses
rare exact terms (a method name, a dataset, a gene symbol), which is precisely
the vocabulary a paper corpus is full of. On the WANDS benchmark a tuned hybrid
reaches 0.7497 nDCG against 0.6983 for BM25 and 0.6953 for pure vector — a 7.4%
lift over *either* leg alone, which is the point: the legs are complementary, not
redundant.

**Decision: fuse with Reciprocal Rank Fusion, not weighted score blending.** [P][B]
RRF operates on ranks, not scores. BM25 scores are unbounded and corpus-dependent;
cosine similarity is bounded in [-1, 1]. Any weighted sum of the two needs a
normalisation constant that must be re-tuned whenever the corpus changes — a
hidden hyperparameter that silently rots exactly like a stale chunk id. RRF has no
such constant. (Cormack, Clarke & Büttcher, SIGIR 2009.)

**Decision: retrieve ~4×`top_k` candidates before reranking (min 20, max 200).** [B][E]
Published practice puts the cross-encoder input pool at 50–200. Too small and the
reranker cannot rescue a buried positive; too large and you pay cross-encoder
latency on a tail that will never surface. `4× top_k` sits in that band for our
`top_k` of 5–20 and is set in one place (`api_S.py`), so it is a tunable, not a
scattered magic number.

**Decision: keep `bm25_score`, `vector_score` and `rerank_score` as separate
fields.** [E] Collapsing them into one number makes it impossible to attribute a
metric change to the lexical leg, the dense leg, or the reranker. Three columns
cost nothing and turn "nDCG went down" into a diagnosable event.

### Stage 3 — MCP tool layer (Sude)

Three tools: `search_papers`, `get_citation`, `summarize_section`.

**Decision: real MCP server as a thin adapter over a FastAPI service.** [D][E]
The tool *logic* is HTTP; the MCP layer only translates protocol. This means the
logic is testable with `TestClient` without speaking MCP, the eval gate can call
it directly instead of spawning an MCP session per query, and a protocol change
is a one-file change.

**Decision: tools never raise across the MCP boundary — they return `ToolError`
with a `retryable` flag.** [E] An agent that receives a stack trace cannot
recover; one that receives `code="no_results", retryable=True` can broaden its
query and try again. This is what makes the researcher→retry edge in stage 4
possible at all.

**Decision: `get_citation` returns a verbatim `quote` plus its character span.** [P][B]
Groundedness means every factual claim traces back to retrieved context. A
citation that names a paper but not a span is not checkable — the judge would
have to verify a claim against a whole document, which is both expensive and
where long-context failures bite hardest. Returning the span makes verification a
short, local comparison. Current guidance is explicit that claims which cannot be
traced to retrieved context should be flagged or stripped; you can only do that
if the trace target is small.

### Stage 4 — The agent graph (Sude)

`researcher → writer → editor →` (conditional) `→ researcher | END | flag_for_human`,
capped at 3 revisions.

**Decision: small `top_k` (5) and spans-not-documents into the writer's context.** [P][B]
This is the context-rot constraint, and it is the one most teams get wrong by
assuming a large context window means they can skip retrieval quality. Chroma's
2025 evaluation of 18 frontier models found *every* model degrades as input length
grows. The positional pattern is worse than a smooth decay: accuracy is highest
when relevant information sits at the very start or very end of the context and
drops by **more than 30%** when it sits in the middle — and that U-shape only
holds while the context is under ~50% full; past that, recent tokens win outright
and early tokens lose. Semantically similar distractors make it worse, which is
exactly what a top-50 retrieval from a single-topic paper corpus produces.
Consequence for us: retrieve wide (200 candidates), *deliver narrow* (5 spans),
and never pad the writer's context with "probably useful" chunks. The reranker's
job is not to add recall; it is to make a 5-slot context worth having.

**Decision: revision cap of 3, then `flag_for_human`.** [E][D] An editor→researcher
loop with no cap is an unbounded cost and an unbounded latency tail in a system
that is also a CI dependency. Three is a decision, not a finding — recorded here
so it can be challenged with data once we can measure how many revisions actually
converge.

**Amended: the revision edge is two edges, routed by error depth.** [P]
*(September 2026 — supersedes the single revision arrow drawn in §1. Rationale and
sources in `docs/research_notes_S.md` §2.1; implemented in
`contracts/judge_J.py`.)* The self-correction literature is blunt that *intrinsic*
self-correction — a model revising its own work with no external signal — makes
output worse rather than merely failing to help. Our editor already escapes that,
because it routes on a judge verdict computed against retrieved sources. But the
same literature attaches a second condition we had not designed for: revision
reliably repairs **shallow** errors and reliably compounds **deep** ones, where
deep means the evidence needed to fix the claim is not in the writer's context at
all. A single undifferentiated revision edge is therefore wrong in one direction
or the other on every pass.

So the editor routes on `FaithfulnessViolation.kind`:

| Kind | Depth | Routes to |
|---|---|---|
| `missing_citation`, `overstated_certainty`, `misattributed_citation` | shallow | **writer** — rewrite |
| `unsupported_claim`, `inferential_leap`, `contradicts_source` | deep | **researcher** — re-retrieve |

A writer asked to repair an `unsupported_claim` without new evidence has exactly
two moves available: delete the claim, or invent support. Routing deep violations
back to retrieval removes that trap structurally rather than by prompt wording,
which is the same reason `passed` is rubric-derived rather than self-reported.

Open, not decided: whether the cap of 3 should become per-branch, since three
re-retrievals is a different cost profile from three rewrites. Left single until
the trajectory metrics below can say how many of each actually converge.

**Amended: `flag_for_human` also fires on low judge confidence.** [P] As
originally drawn, human escalation was purely a timeout — it happened when the
revision budget ran out. `FaithfulnessVerdict.confidence` makes it an uncertainty
signal as well, checked *before* `passed`, so a draft that passed with an unsure
judge is not treated as indistinguishable from a confident pass. Ships disabled
(`escalation_confidence = 0.0`) until the judge is calibrated against Buse's
qrels, because a threshold on an uncalibrated number is not defensible.

**Decision: explicit permission to abstain.** [P][B] Many hallucinations happen
because the model was never allowed to refuse. Recent work on grounded response
and abstention reports refusal rate rising 0.767 → 0.828 with hallucination rate
falling 0.128 → 0.083 — a large reduction bought by making "the evidence does not
answer this" a first-class output. Our `EvalQuery.intent` includes an
`unanswerable` class precisely so the held-out set can measure whether we abstain
when we should, instead of only measuring performance on answerable questions.

**Decision: temperature by role, not one global setting.** [D][E]
- Judge: **0.0**. A judge that varies run-to-run cannot support a promote/reject
  decision, and reproducibility is the whole point of the gate.
- Editor: **0.0**. It applies a rubric; creativity here is a bug.
- Writer: **0.3**. Non-zero for readable prose, low because the failure mode we
  care about is invention, not dullness. Every claim is verified downstream
  regardless, so this is a legibility knob, not a safety one.
- Researcher (query reformulation): **0.7**. This is the one place where variety
  pays — a reformulation that explores different vocabulary is what rescues a
  failed retrieval. It is safe to be creative here because its output is a
  *query*, never text that reaches the draft.

**Amended: not implementable on Claude 5 models.** [E] *(September 2026.)*
Sampling parameters — `temperature`, `top_p`, `top_k` — are removed on the Claude 5
generation and return a 400, so the real client sends none; the settings above
remain the record of intent and apply only to older models. Two consequences:
the gate's reproducibility rests on the deterministic stub backend, as it already
did; and on a real model, run-to-run judge variance must be **measured** — grade
the same items more than once and report agreement — rather than assumed to be
zero. That measurement belongs next to κ in the calibration report.

**Decision: parallelise independent tool calls.** [B] Retrieval is roughly 35% of
RAG time-to-first-token and its p95 can run 64× the median — the tail, not the
median, is what users feel. Independent tool calls run concurrently save on the
order of 150ms each, and sequential blocking I/O inside an agent loop is the
classic source of hundreds of wasted milliseconds. Concretely: the writer's
`get_citation` calls for N chunks have no data dependency on each other and must
be issued together.

### Stage 5 — Judge (Sude)

One judge, two uses: grade retrieved chunks (feeds stage 7), check drafts against
sources (feeds the editor).

**Decision: graded relevance 0–3, not binary.** [P][E] nDCG needs gradations to
be more than a hit-rate, and DPO pairs need a *margin* to filter on. Binary labels
throw away both. Scale matches the TREC convention Buse's qrels use, so judge
verdicts and human labels are directly comparable — which is what makes the next
decision possible.

**Decision: calibrate the judge against Buse's human qrels and report Cohen's
kappa; rework the rubric if kappa < 0.5.** [P][B] This is the guardrail on the
guardrail. LLM judges can reach ~80% agreement with humans — comparable to
human-to-human agreement (Zheng et al., MT-Bench, 2023) — but 2026 large-scale
evaluations are blunt about the failure modes: no judge is uniformly reliable
across benchmarks, judge rankings shift by up to 14 positions depending on the
benchmark, and high test–retest reliability (>0.95) coexists with severe position
bias. "Reliability without validity" is the exact trap: a judge that agrees with
*itself* while disagreeing with humans will happily promote a worse reranker.
Kappa against human labels is the only thing that catches it, and Buse's qrels
already give us the human labels for free.

**Decision: randomise presentation order and score chunks independently.** [P][B]
Position bias is measured, not hypothetical — reported at ~40% inconsistency for
GPT-4-class judges in pairwise settings, and found to be systematic rather than
random across ~150,000 evaluation instances and 15 judges. Two mitigations:
grade each chunk on an absolute 0–3 scale rather than pairwise where possible, and
where order exists, shuffle it deterministically by a seed we log.

**Decision: decompose drafts into claims, verify each against its cited span.** [P][B]
The RAGAS approach — split the answer into atomic claims, check each against
context — is what makes `citation_precision` and `coverage` computable rather than
impressionistic, and it keeps each verification short, which sidesteps the
long-context degradation in the stage-4 note.

**Decision: `FaithfulnessVerdict.passed` is derived by the rubric, not
self-reported by the model.** [E] The editor agent routes on this boolean. A model
asked "did you pass?" is being asked to grade its own work; the rubric in
`docs/editorial_rubric_S.md` turns counted violations into a decision.

**Decision: every verdict carries `JudgeMeta` (model, prompt version, temperature).** [E]
Without it, a metric shift cannot be attributed to the system versus a judge-prompt
edit. This is the single cheapest piece of experimental hygiene in the project.

### Stage 6–7 — Preference pairs and DPO (Sude → Buse)

**Decision: DPO rather than a reward model + PPO.** [P] Direct Preference
Optimization (Rafailov et al., 2023) fits the preference objective directly, with
no separate reward model to train, host and debug. For a two-person project, the
removed moving part *is* the argument.

**Decision: keep `margin` on every pair and drop near-ties at training time.** [P][E]
Judge-built preference data is noisy, and pairs with a thin grade gap are where
the noise concentrates — a 2-vs-3 pair often reflects judge variance rather than a
real preference. Filtering on margin is cheaper than training through the noise.
Retaining the field rather than pre-filtering lets Buse tune the threshold
without regenerating the dataset.

**Ownership note:** Sude's judge emits `PreferencePair` JSONL; Buse trains the
cross-encoder. Sude never imports the reranker — the `Reranker` Protocol is the
only contact point.

### Stage 5 (cont.) — The CI gate (Sude)

**Decision: the gate refuses to compare runs whose `corpus_version`,
`embedding_model` or `reranker_version` stamps differ.** [E] Otherwise a "+4 nDCG"
that was really a corpus change gets promoted, and the gate becomes a rubber stamp.
This is the mechanical version of "control your variables".

**Decision: strict dataset loading — a malformed row fails the gate.** [E] The
tempting alternative (skip bad rows) silently shrinks the eval set, which makes
the gate *easier* to pass exactly when the data is degrading.

**Amended: the gate compares with a paired bootstrap, not two point estimates.**
[B] *(September 2026 — `eval/metrics/significance_S.py`; rationale in
`docs/research_notes_S.md` §2.4.)* Comparing `candidate > baseline` on two means
ignores that the measurement is itself noisy: per-query variance in nDCG is far
larger than the effects we are trying to detect, so a bare threshold comparison
promotes noise roughly half the time it fires on a no-op change. Three parts:

- **Paired.** Both runs score the *same* queries and we bootstrap the per-query
  *difference*, which cancels query difficulty — the dominant variance term. This
  is the cheapest sensitivity we can buy: it makes the same labels from Buse
  detect a substantially smaller true effect. `align()` refuses to compare runs
  covering different query sets rather than silently intersecting them, for the
  same reason the loader is strict.
- **Interval, not point.** A change counts only when the 95% CI excludes zero.
  `PairedResult.direction` returns `inconclusive` otherwise, and inconclusive is
  a first-class answer — a gate that cannot say "I don't know" is a gate that
  promotes noise.
- **Seeded.** The resampling seed is an explicit argument and is logged. A
  promote/reject decision that cannot be reproduced cannot be appealed.

`minimum_detectable_effect(n, sd)` turns "the gate went green" into "the gate can
see changes of at least X", which is the sentence that belongs in the results
section rather than a bare metric delta. Stdlib only — no numpy, no scipy —
because a gate with an install step breaks on the one push where it matters.

Consequence for Buse: the eval-set size is now a quantity with a right answer
rather than a guess, and she needs it before the qrels are finished. See
`docs/contracts_J.md` open question 4 for the table.

**Decision: `test` split never touches prompt or reranker selection; `dev` does.** [P][E]
Standard held-out discipline, enforced mechanically by
`tests/test_no_eval_leakage_J.py` rather than by good intentions.

**Decision: the CI backend is a deterministic stub, not a paid API.** [E] A gate
that costs money on every push is a gate people learn to skip, and a
non-deterministic judge makes a promote/reject decision unreproducible. The stub
tests the plumbing on every push; scheduled runs against a real model produce the
quality numbers.

### Cross-cutting — Observability (Sude)

**Decision: instrumentation is optional at import *and* at runtime, and never
raises.** [E] `@traced` costs one attribute lookup when disabled and degrades to
structured logs when Langfuse is unconfigured. This is what makes it safe to
decorate *every* tool, judge call and node — which is the only way to get the
per-stage p50/p95/p99 breakdown that production RAG guidance calls for. You cannot
find a bottleneck you did not instrument before you went looking.

---

## 3. Non-functional budgets

Numbers to design against and then measure. Sources for the shape of the budget
are cited; the specific split is ours to defend or revise once instrumented.

**Latency (per query, p95 target ≤ 3.5s to first token):**

| Stage | Budget | Note |
|---|---|---|
| Embed query | 50 ms | Single short string |
| Hybrid retrieval (both legs, parallel) | 300 ms | Legs run concurrently, so cost is the slower one |
| Fusion (RRF) | <5 ms | Pure arithmetic on ranks |
| Cross-encoder rerank (20–200 pairs) | 400 ms | The adjustable knob — pool size is the lever |
| Writer prefill + first token | ~2.5 s | Dominated by model, not us |

Retrieval is ~35% of TTFT with a p95 up to 64× median, so the gate should assert
on **p95, not mean** — a mean-based SLA hides exactly the failure users notice.

**Context budget (writer):** 5 spans, section-sized. Justified by the >30%
mid-context accuracy drop and the sub-50%-fullness rule in §2 stage 4. If a draft
needs more evidence, the answer is another retrieval round (the researcher edge),
not a longer prompt.

**Bottleneck watchlist**, in the order they will bite:
1. Cross-encoder reranking on a large pool — mitigate by pool size, and by an
   adaptive budget if p95 slips.
2. Sequential `get_citation` calls in the writer — must be concurrent.
3. Judge calls in the eval gate — embarrassingly parallel; batch them.
4. Cold vector-store connections per request — pool and reuse.

---

## 4. Duties, and who blocks whom

**Buse blocks Sude on:** the eval query set and qrels (without them the gate has
nothing to gate on), and the retrieval service (until it lands, Sude's stack runs
on `StubRetrieval`, whose numbers are explicitly not publishable —
`readiness()` reports `degraded` so this can never be mistaken for a real run).

**Sude blocks Buse on:** preference pairs (no pairs, no DPO training) and the MCP
tool schemas (already drafted in `contracts/mcp_tools_J.py`).

**Neither blocks the other on:** everything else. That is the point of the seam —
`StubRetrieval` and the strict contracts exist so both tracks can run end-to-end
alone and only integrate at four agreed places.

Build order (unchanged from both READMEs):
1. Plain RAG + MCP tools, manually evaluated — **joint**
2. Ingestion/vector DB (Buse) ∥ judge + preference pairs (Sude)
3. CI/CD gate (Sude) on thresholds Buse defines
4. MCP tools finalised + observability, then agent graph (Sude)

---

## 5. Open questions for Buse

1. Confirm the reranker split: Buse trains, Sude supplies pairs. (Files are
   suffixed `_B` but Sude's README assigns DPO to her.)
2. `observability/tracing_S.py` is `_S` but Buse's README assigns observability to
   her. Sude has implemented it; rename to `_J` if you would rather co-own it.
3. Confirm the embedding model, so `RetrievalResponse.embedding_model` is not free
   text, and confirm `corpus_version` bumping is manual.
4. Do the qrels cover `intent="unanswerable"` queries? Without them we cannot
   measure abstention, which is a headline claim of the design.

---

## 6. Reading list

Ordered by what it unblocks. **Tier 1** is the minimum to defend this design in
the report; tier 2 is depth for your own track; tier 3 is the frontier we cite as
"future work" rather than implement.

### Tier 1 — read before writing the report (8 papers)

| Paper | arXiv / venue | Which decision it backs |
|---|---|---|
| Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020) | [2005.11401](https://arxiv.org/abs/2005.11401), NeurIPS | The premise of the whole system |
| Karpukhin et al., *Dense Passage Retrieval for Open-Domain QA* (2020) | [2004.04906](https://arxiv.org/abs/2004.04906), EMNLP | The dense leg of stage 2 |
| Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion Outperforms Condorcet…* (2009) | SIGIR 2009 | RRF over weighted score blending |
| Nogueira & Cho, *Passage Re-ranking with BERT* (2019) | [1901.04085](https://arxiv.org/abs/1901.04085) | Why a cross-encoder reranks a candidate pool |
| Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (2023) | [2307.03172](https://arxiv.org/abs/2307.03172), TACL | The >30% mid-context drop → small `top_k`, spans not documents |
| Rafailov et al., *Direct Preference Optimization* (2023) | [2305.18290](https://arxiv.org/abs/2305.18290), NeurIPS | DPO instead of reward model + PPO |
| Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023) | [2306.05685](https://arxiv.org/abs/2306.05685) | ~80% judge/human agreement — and position bias |
| Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023) | [2309.15217](https://arxiv.org/abs/2309.15217) | Claim decomposition → faithfulness, citation precision |

### Tier 2 — depth, by track

**Retrieval and evaluation (Buse's track, but read §nDCG yourself):**
- Järvelin & Kekäläinen, *Cumulated Gain-Based Evaluation of IR Techniques* (TOIS 2002) — the actual definition of nDCG, and why graded relevance beats binary.
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009) — what BM25's `k1`/`b` actually do.
- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot IR* (2021) — [2104.08663](https://arxiv.org/abs/2104.08663). The evidence that no single retriever wins everywhere; the case for hybrid.
- Khattab & Zaharia, *ColBERT* (2020) — [2004.12832](https://arxiv.org/abs/2004.12832). Late interaction: the middle ground between bi- and cross-encoders if reranking latency becomes the bottleneck.
- Formal et al., *SPLADE* (2021) — [2107.05720](https://arxiv.org/abs/2107.05720). Learned sparse retrieval — a possible upgrade to the BM25 leg.

**Generation, grounding and judging (your track):**
- Gao et al., *Enabling LLMs to Generate Text with Citations* (ALCE, 2023) — [2305.14627](https://arxiv.org/abs/2305.14627). **The closest paper to what your writer/editor agents do.** Read this one properly.
- Rashkin et al., *Measuring Attribution in Natural Language Generation Models* (2021) — [2112.12870](https://arxiv.org/abs/2112.12870). The AIS framework: the formal definition of "this claim is supported by this source", which is what your rubric is operationalising.
- Min et al., *FActScore* (2023) — [2305.14251](https://arxiv.org/abs/2305.14251). Atomic-fact decomposition; the method behind your claim-level verification.
- Saad-Falcon et al., *ARES* (2023) — [2311.09476](https://arxiv.org/abs/2311.09476). Automated RAG evaluation with human-label calibration — directly relevant to the kappa guardrail.
- Kadavath et al., *Language Models (Mostly) Know What They Know* (2022) — [2207.05221](https://arxiv.org/abs/2207.05221). Calibration and self-knowledge: the basis for principled abstention.

**Agent design (your track):**
- Yao et al., *ReAct: Synergizing Reasoning and Acting* (2022) — [2210.03629](https://arxiv.org/abs/2210.03629). The researcher node's loop.
- Madaan et al., *Self-Refine* (2023) — [2303.17651](https://arxiv.org/abs/2303.17651). The writer→editor→revise edge, and its limits.
- Shinn et al., *Reflexion* (2023) — [2303.11366](https://arxiv.org/abs/2303.11366). Verbal feedback as the revision signal — the justification for the editor passing violations back, not just a boolean.
- Asai et al., *Self-RAG* (2023) — [2310.11511](https://arxiv.org/abs/2310.11511). Learned retrieve/critique tokens — the ambitious version of what our graph does with explicit edges.

**Preference optimisation (context for Buse's DPO work):**
- Ouyang et al., *InstructGPT* (2022) — [2203.02155](https://arxiv.org/abs/2203.02155). The RLHF baseline DPO replaces.
- Ethayarajh et al., *KTO* (2024) — [2402.01306](https://arxiv.org/abs/2402.01306) and Hong et al., *ORPO* (2024) — [2403.07691](https://arxiv.org/abs/2403.07691). Alternatives worth a sentence in related work.
- Meng et al., *SimPO* (2024) — [2405.14734](https://arxiv.org/abs/2405.14734). Reference-free; relevant if the DPO run is memory-bound.

### Tier 3 — 2026 frontier, for related work and future work

The recent items in §Sources below, especially the LLM-as-judge reliability
audits (they are the strongest argument for our kappa guardrail), the context-rot
follow-ups, and the adaptive rerank-budget work (the principled fix if p95 slips).

> Note on provenance: tier 1 and 2 are established papers cited from their canonical
> IDs — verify each link resolves before it goes in the bibliography. The 2026
> items in §Sources came from a live search and include vendor engineering blogs
> alongside arXiv preprints; the tags in §2 mark which is which, and only **[P]**
> items belong in the report's citation list without qualification.

---

## Sources

- [Lost in the Middle / positional degradation and context rot](https://www.morphllm.com/context-rot)
- [Diagnosing and Mitigating Context Rot in Long-horizon Search](https://arxiv.org/pdf/2606.29718)
- [Intelligence Degradation in Long-Context LLMs](https://arxiv.org/pdf/2601.15300)
- [Hybrid Search: BM25, Vector & Reranking Reference 2026](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026)
- [From BM25 to Corrective RAG: Benchmarking Retrieval Strategies](https://arxiv.org/pdf/2604.01733)
- [Hybrid Search for RAG: Combining BM25 and Dense Vector Search](https://denser.ai/blog/hybrid-search-for-rag/)
- [Reliability without Validity: Large-Scale Evaluation of LLM-as-a-Judge](https://arxiv.org/html/2606.19544v1)
- [The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge Evaluation](https://arxiv.org/pdf/2606.13685)
- [LLM-as-Judge Best Practices 2026: Calibration, Bias, and Cost](https://futureagi.com/blog/llm-as-judge-best-practices-2026/)
- [GRACE: RL for Grounded Response and Abstention under Contextual Evidence](https://arxiv.org/pdf/2601.04525)
- [CiteGuard: Faithful Citation Attribution via Retrieval-Augmented Validation](https://arxiv.org/pdf/2510.17853)
- [RAG Groundedness Evaluation Guide](https://www.openlayer.com/blog/measuring-rag-groundedness-complete-evaluation-guide)
- [Cascading Hallucination in Agentic RAG: the CHARM framework](https://arxiv.org/pdf/2606.04435)
- [A Latency Budget for Production RAG](https://www.technovice.net/post/latency-budget-production-rag)
- [Less can be More: Relieving RAG Bottlenecks via Evidence Frontloading and Pressure-Adaptive Budgeting](https://arxiv.org/abs/2608.25115)
- [Designing low-latency AI agents through reranker optimization](https://decagon.ai/blog/designing-low-latency-ai-agents-through-reranker-optimization)
- [AI Agent Latency Budgets: 6-Tier Framework](https://www.kunalganglani.com/blog/ai-agent-latency-optimization-budget)
