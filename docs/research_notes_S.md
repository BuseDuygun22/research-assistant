# Research notes: agent architecture, judging and evaluation (2026)

Status: literature sweep by Sude, September 2026. Scope is Track S only — the
agent graph, the judge, the MCP layer, the eval gate and observability. Nothing
here proposes changes to ingestion, retrieval or the reranker (Buse's track),
except where a contract is shared and therefore joint.

**Implementation status (16 September 2026).** Items 1–3 of §5 have landed, along
with all three contract changes in §4; see the status column in §5. The three
contract edits are additive with backward-compatible defaults and are flagged for
Buse in `docs/contracts_J.md` open question 5 — she can still veto them without
breaking anything built against the previous shapes. Items 4–6 are not started,
because each needs a module that does not exist yet.

Companion to `docs/architecture_J.md`, which this either confirms or amends.
Evidence tags follow that document's convention:

| Tag | Means |
|---|---|
| **[P]** | Peer-reviewed or arXiv paper |
| **[B]** | Published benchmark, measured report, or practitioner guidance |
| **[E]** | Engineering constraint we can demonstrate ourselves |
| **[D]** | Team decision — no external evidence |

> Provenance warning, same as the parent doc: items below drawn from vendor
> engineering blogs are tagged **[B]**, not **[P]**, and should not enter the
> report's citation list without qualification. Several 2026 arXiv items are
> preprints; verify each resolves before the bibliography is frozen.

---

## 0. The frame

An agent architecture is three decisions in a loop: what the model sees
(context), who checks the output (verification), and when the loop stops
(control). Nearly every advance since 2023 is a better answer to one of the
three. `architecture_J.md` answers all three explicitly, which is why most of
what follows is refinement rather than redesign.

---

## 1. Confirmed by the 2026 literature — do not churn

**Small graph over a swarm.** [B] The strongest 2026 correction to multi-agent
enthusiasm: a single agent that can hold the task in context beats a team that
must coordinate, and finite context is the only honest reason to split. Reported:
a single agent matched or beat multi-agent systems on 64% of benchmarked tasks
with the same tools, multi-agent buying ~2.1 percentage points at roughly double
the cost; production failure-rate surveys run 41–86.7%. Our three nodes with
distinct jobs and one shared state object is the shape that survived. Consequence:
resist adding a fourth agent. Add a tool or an edge instead.

**External verification over self-critique.** [P] See §2.1 — confirmed, but with
a condition we have not yet designed for.

**Retrieve wide, deliver narrow.** [P][B] The context-rot argument holds. The 2026
refinement is mechanistic: semantically similar distractors, not position alone,
are what poison a small context — which is precisely what a single-topic paper
corpus yields. Strengthens rather than weakens the §2-stage-4 argument.

**Abstention as a first-class output.** [P] Now supported by RL methods (GRACE)
and dedicated benchmarks for knowing when not to answer. `EvalQuery.intent`'s
`unanswerable` class remains the right instrument.

**Machine-readable tool errors.** [B] Current MCP guidance converged
independently on nearly our exact `ToolError`. See §4.1 for the two fields it
goes further on.

---

## 2. Where the literature has moved past the current design

### 2.1 Self-correction is conditional on error depth [P]

Established and replicated: *intrinsic* self-correction — a model revising its own
work with no external signal — degrades output on reasoning-style tasks rather
than merely failing to help. The bottleneck is feedback generation: models cannot
reliably judge the correctness of their own prior response.

Our editor escapes this, because it routes on a judge verdict computed against
retrieved sources — external grounding, not self-assessment. That remains sound.

The 2026 addition is a second condition. The **error depth hypothesis**
distinguishes shallow errors (surface-level; revision fixes them) from deep errors
(the underlying reasoning or evidence is wrong; revision compounds them). The
associated **accuracy-correction paradox** is that stronger initial drafts benefit
*less* from revision — so a uniform "revise up to 3 times" penalises exactly the
good cases.

**Amendment proposed to `architecture_J.md` §2 stage 4.** The revision edge should
be two edges, routed on `FaithfulnessViolation.kind`, which already exists:

| Kind | Depth | Route to | Reasoning |
|---|---|---|---|
| `missing_citation` | shallow | writer | Attach the citation; evidence is present |
| `overstated_certainty` | shallow | writer | Hedge the claim |
| `misattributed_citation` | shallow | writer | Swap to the correct chunk |
| `unsupported_claim` | **deep** | researcher | Evidence absent from context — rewriting cannot fix it |
| `contradicts_source` | **deep** | researcher | Re-retrieve, or drop the claim |

A writer asked to repair an `unsupported_claim` without new evidence has two
available moves: delete the claim, or fabricate support. Routing deep violations
back to retrieval removes that trap structurally rather than by prompt wording.

Secondary consequence: the revision cap should arguably be per-branch. Three
re-retrievals is a different cost profile from three rewrites.

### 2.2 Route on calibrated confidence, not a boolean [P]

`FaithfulnessVerdict.passed` is a bool, and the editor routes on it. The 2026
direction — CalVerT is the cleanest instance — places *calibrated* verifier
confidence into agent state and conditions actions on it, where calibrated means
temperature-scaled so a stated 0.7 corresponds to ~70% empirical correctness
rather than an uninterpretable raw score. External verifier signal outperforms
self-critique; calibration is what makes it usable for routing rather than
merely informative.

A boolean forces identical behaviour on "barely passed" and "passed decisively".
Confidence unlocks the third route the design already wants:
**`flag_for_human` on low confidence, not only on an exhausted revision budget.**
As designed, human escalation is a timeout; it should be an uncertainty signal.

`RelevanceVerdict` already carries `confidence`; `FaithfulnessVerdict` does not.
Closing that asymmetry is a joint contract change (§4).

### 2.3 A jury of small judges beats one large judge, and costs less [P][B]

Our judge is a single model guarded by a kappa check. The stronger current design
is a panel of several smaller, *diverse* models scoring independently and
aggregated (PoLL; RoPoLL adds robust aggregation).

Reported: a panel of smaller models outperformed a single GPT-4-class judge at
roughly **7–8× lower cost**, with the cost/accuracy knee near **N ≈ 3**.

The argument that matters for us is not cost, it is validity. `architecture_J.md`
already names the "reliability without validity" trap — a judge that agrees with
itself while disagreeing with humans. A single model shares its idiosyncratic
biases with itself by construction; independent diverse judges cannot. A jury is
the structural mitigation for position and style bias that no amount of
prompt-engineering a single judge will buy.

**Calibration practice has also firmed up** [B]: 200–500 hand-labelled examples,
2–3 human labellers each, and kappa read *alongside the confusion matrix*, class
prevalence and the actual disagreements — not as a standalone score. This matters
for our graded 0–3 scale, where grade 0 dominates and prevalence distorts kappa.
Our "rework if kappa < 0.5" rule survives as a tripwire but should not be the only
read.

### 2.4 The CI gate needs error bars [B]

The gate as designed compares metrics to thresholds and promotes or rejects.
Nothing in it accounts for the measurement being noisy. Three consequences:

1. **Point estimates mislead.** Current practice is bootstrap confidence
   intervals (~1000 resamples), treating a change as real only when the 95% CI
   **excludes zero**. Without this the gate promotes noise.
2. **Eval-set size bounds what is detectable.** ~200 queries to resolve a 3–5%
   difference; beyond ~500, diminishing returns absent distinct sub-workloads.
   *This is a number Buse needs before finishing qrels* — it is a blocking
   dependency, and it belongs in §5 of `architecture_J.md` as a fifth open
   question.
3. **Multiple comparisons.** A two-sided 5% gate is wrong ~5% of the time when
   nothing has changed. Run per-PR, that is a steady trickle of false regressions,
   and a gate people learn to skip — the same failure mode the doc already cites
   for a gate that costs money.

**Highest-value change: paired comparison.** Run baseline and candidate over the
*same* queries and bootstrap the per-query *difference*. Per-query difficulty
variance cancels, so far smaller true effects become detectable from the same
labelled set. Cost: a few lines in `eval/run_gate_S.py`. Effect: it raises the
sensitivity of every label Buse produces, which makes it the cheapest respect we
can pay her annotation time.

This composes with — and does not replace — the existing stamp check on
`corpus_version` / `embedding_model` / `reranker_version`.

### 2.5 Evaluate the trajectory, not only the final answer [P][B]

Our gate measures endpoints: nDCG@5, MRR, citation precision, faithfulness. The
2026 shift is toward step-level process evaluation (process reward models for
agents; step-level trajectory evaluation and aggregation), on the grounds that a
system can reach an acceptable answer by a broken path that will fail differently
next week. Agentic RAG surveys name the same gap: evaluation of agentic RAG
"remains nascent", with explainability of decision traces an open dimension.

Questions our endpoint metrics cannot answer:

- Did the researcher's reformulation *improve* retrieval, or only spend a round?
- What share of drafts pass on revision 1 vs 3? — the data that converts our
  "3 is a decision, not a finding" into a finding.
- What fraction of tool calls return `ToolError`, and does the agent recover?

We are one step from this: `observability/tracing_S.py` already instruments every
node. The work is turning traces into *gated metrics* rather than debug output.
Suggested trajectory metrics for the gate: revision-count distribution, retrieval
improvement per researcher round (nDCG before vs after reformulation), tool-error
and recovery rate, abstention rate split by `intent`.

### 2.6 Offload, do not compact [B]

Current context-engineering consensus runs slightly against instinct: under prompt
caching, retaining full history frequently beats summarising it on cost, latency
*and* recall simultaneously. Compaction is lossy and irreversible, and 2026 work
documents "governance decay" — constraints stated early being silently dropped by
compaction in long-horizon agents. The replacement rule is **do not delete,
relocate**: offload detail to files or a store, keep a pointer in context,
retrieve just in time.

This does **not** touch the 5-span writer budget, which is evidence selection and
stays as specified. It applies one level up, to state carried across revisions in
`agents/state_S.py`. When `AgentState` accumulates drafts and verdicts the
temptation will be to summarise history to keep prompts small. Don't — keep prior
drafts and violations addressable by id, and pass the writer only the current
violations. The same discipline as the span budget, applied to agent state.

---

## 3. Frontier — report as future work, do not implement

**The retrieval policy can be trained rather than prompted.** [P] Search-R1 and
ConvSearch-R1 train query reformulation against retrieval reward (rank-incentive
shaping: large reward for reaching the top 10, proportionally smaller for 11–100)
instead of relying on a prompted, high-temperature rewrite. ConvSearch-R1 removes
the need for external rewrite supervision entirely by learning from retrieval
signal.

Why this is ours to claim: **we already emit the reward signal.** The judge
produces graded 0–3 relevance per chunk — the same shape of signal these methods
shape a rank-based reward from. The preference-pair pipeline built for the
reranker generalises to the reformulator with no new labelling.

Not for a two-person project. But the report's future-work section can honestly
say that our judge pipeline is not reranker-specific — it is a retrieval-policy
reward, which is the direction the field has taken.

**Motivation for the report's opening.** [B] 2026 evaluations of deep research
agents found citation quality and factual accuracy to be the *weakest* axes, with
the best system reaching roughly 65% citation quality and 68% factual accuracy.
The problem this project picked is unsolved at the frontier, not a teaching
exercise, and the number says so.

---

## 4. Proposed contract changes — JOINT, for Buse's agreement

These touch `contracts/*_J.py` and are proposals, not edits.

### 4.1 `ToolError` gains `remediation` and `partial` [B]

Current MCP guidance recommends error payloads carrying
`code, message, retryable, partial, remediation`. `remediation` tells the agent
*what to do*, not merely that a retry is permitted; `partial` lets
`summarize_section` return four of five chunks rather than failing whole. Our
existing rationale for `ToolError` already makes this argument — the field list
simply stops two fields short of it.

Related, same source: tool schemas carrying concrete usage examples raised
accuracy on complex parameter handling from 72% to 90%. Cheap for three tools.
(Schema bloat — 100–500 tokens per definition, 30–50% of context in large
multi-server setups — is not our problem at three tools, but is the reason to keep
it at three.)

### 4.2 A violation kind for inferential leaps [B]

2026 deep-research evaluations attribute residual citation errors *primarily* to
improper inferential linking and inaccurate paraphrasing: the source supports A,
the draft asserts B, and B is a plausible but unsupported leap from A. Under the
current taxonomy this lands in `unsupported_claim`, which is also where "no source
at all" lands. Different failures, different fixes — and the literature says the
inferential one is the common case. Proposed: `inferential_leap`, classified
**deep** under §2.1 routing.

### 4.3 `FaithfulnessVerdict` gains `confidence` [P]

Per §2.2, so the editor can escalate on uncertainty rather than only on an
exhausted budget. `RelevanceVerdict` already has the field; this closes the
asymmetry. `passed` stays rubric-derived — this adds a signal, it does not move
the decision back inside the model.

---

## 5. Priority

| # | Change | Cost | Status | Where |
|---|---|---|---|---|
| 1 | Route revisions on violation depth (§2.1) | Small | **Landed** | `contracts/judge_J.py` — `DEEP_VIOLATION_KINDS`, `is_deep`, `revision_route()` |
| 2 | Paired bootstrap CIs in the gate (§2.4) | Small | **Landed** | `eval/metrics/significance_S.py` |
| 3 | `FaithfulnessVerdict.confidence` + escalate on uncertainty (§2.2, §4.3) | Small | **Landed** | `contracts/judge_J.py`, `Settings.escalation_confidence` |
| 4 | Judge panel of ~3 small models (§2.3) | Medium | Not started | Blocked: `judge/*_S.py` are empty |
| 5 | Trajectory metrics from existing traces (§2.5) | Medium | Not started | Blocked: no graph to trace yet |
| 6 | RL-trained researcher (§3) | — | Won't do | Future-work section, not code |

Also landed alongside items 1–3:

- `ToolError.remediation` and `.partial` (§4.1), **populated at all five error
  sites** in `mcp_server/api_S.py` rather than left as unused fields. The
  `summarize_section` partial path is a real behaviour change: previously, asking
  for five chunk ids where two did not exist returned a summary of the other
  three with no indication anything was missing, which would let the writer cite
  a section it never saw summarised. It now returns the summary *and* a
  `partial=True` error naming the absent ids.
- `inferential_leap` as a sixth violation kind (§4.2), classified deep.
- 36 tests covering all of the above (`tests/contracts/test_contracts_J.py`,
  `tests/unit/eval/test_significance_S.py`), including the negative property that
  matters most: two runs drawn from the same distribution come back
  `inconclusive` rather than promoting.

**Deferred, and why.** Items 4 and 5 are not judgement calls about value — they
are blocked on modules that do not exist. A judge panel needs a judge; trajectory
metrics need a graph to produce trajectories. Both should be built *into* those
files when they are written rather than retrofitted, which is the same argument
that made item 1 urgent while `graph_S.py` is still empty.

---

## 6. Sources

Agent architecture and multi-agent trade-offs:
- [SoK: Agentic Retrieval-Augmented Generation — Taxonomy, Architectures, Evaluation](https://arxiv.org/pdf/2603.07379)
- [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](https://arxiv.org/abs/2501.09136)
- [Deep Research: A Survey of Autonomous Research Agents](https://arxiv.org/pdf/2508.12752)
- [Why Multi-Agent LLM Systems Fail and How to Fix Coordination Issues (2026)](https://www.augmentcode.com/guides/why-multi-agent-llm-systems-fail-and-how-to-fix-them)
- [Context Is the Bottleneck: Why Multi-Agent LLM Systems Fail (MAST)](https://medium.com/@daniel.lh.gordon/context-is-the-bottleneck-why-multi-agent-llm-systems-fail-and-what-mast-teaches-us-b336b9f76e03)
- [Coordination as an Architectural Layer for LLM-Based Multi-Agent Systems](https://arxiv.org/pdf/2605.03310)

Self-correction limits:
- [When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey (TACL)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713/125177/When-Can-LLMs-Actually-Correct-Their-Own-Mistakes)
- [Decomposing LLM Self-Correction: The Accuracy-Correction Paradox and Error Depth Hypothesis](https://arxiv.org/pdf/2601.00828)
- [Confidence v.s. Critique: A Decomposition of Self-Correction Capability](https://arxiv.org/pdf/2412.19513)

Verification, calibration and abstention:
- [CalVerT: Augmenting Agents with Calibrated Verifier Telemetry](https://arxiv.org/pdf/2606.21777)
- [GRACE: RL for Grounded Response and Abstention under Contextual Evidence](https://arxiv.org/html/2601.04525v1)
- [Beyond Fluency: Toward Reliable Trajectories in Agentic IR](https://arxiv.org/html/2604.04269)
- [R2VC: Modular Fact-Checking with Retrieval, Verification, and Confidence Calibration](https://arxiv.org/html/2609.11955)
- [Knowing When Not to Answer: Evaluating Abstention](https://arxiv.org/pdf/2604.14799)

Judging:
- [Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models (PoLL)](https://arxiv.org/pdf/2404.18796)
- [RoPoLL: Robust Panel of LLM Judges](https://arxiv.org/pdf/2606.30931)
- [The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge Evaluation](https://arxiv.org/pdf/2606.13685)
- [How to Calibrate an LLM-as-a-Judge Against Human Labels with Cohen's Kappa](https://oneuptime.com/blog/post/2026-08-31-calibrate-llm-judge-cohens-kappa/view)
- [LLM-as-Judge Best Practices 2026: Calibration, Bias, and Cost](https://futureagi.com/blog/llm-as-judge-best-practices-2026/)

Evaluation methodology:
- [AgentPRM: Process Reward Models for LLM Agents (WWW 2026)](https://dl.acm.org/doi/10.1145/3774904.3792551)
- [Automated Trajectory Evaluation via Step-Level Consequence Reasoning (CRATE)](https://arxiv.org/abs/2608.20797)
- [Measuring all the noises of LLM Evals](https://arxiv.org/pdf/2512.21326)
- [How to Calculate Statistical Significance in LLM Evals](https://futureagi.com/blog/statistical-significance-llm-evals/)
- [Detecting and Correcting Reference Hallucinations in Commercial LLMs and Deep Research Agents](https://arxiv.org/pdf/2604.03173)
- [ReportBench: Evaluating Deep Research Agents via Academic Survey Tasks](https://arxiv.org/pdf/2508.15804)

Context engineering:
- [Context Engineering in 2026: Why We Stopped Compacting Our Agent's Context](https://www.louisbouchard.ai/context-engineering-2026/)
- [Governance Decay: How Context Compaction Silently Erases Safety Constraints](https://arxiv.org/pdf/2606.22528)
- [Code as Agent Harness](https://arxiv.org/pdf/2605.18747)

MCP and tool design:
- [MCP Tool Schema Design Guide 2026 — 7 Principles](https://kansei-link.com/en/insights/mcp-tool-schema-design-guide-2026.html)
- [MCP Tool Schema Bloat: The Hidden Token Tax](https://layered.dev/mcp-tool-schema-bloat-the-hidden-token-tax-and-how-to-fix-it/)

Learned retrieval policies (future work):
- [Search-R1: RL-Enabled Retrieval for LLMs](https://www.emergentmind.com/topics/search-r1)
- [ConvSearch-R1: Query Reformulation with Reasoning via RL (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1349/)
- [A Comprehensive Survey on RL-based Agentic Search](https://arxiv.org/pdf/2510.16724)
- [Less can be More: Evidence Frontloading and Pressure-Adaptive Budgeting (PACE)](https://arxiv.org/abs/2608.25115)
