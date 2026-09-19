# Qrel labeling guide — how to grade a (query, chunk) pair

One question, and only one question, per grade: **how useful is this chunk for
answering this query?** Nothing else belongs in this number.

Decided 2026-09-19. Keeps the scale at 0–3 (TREC-style, same as `RelevanceVerdict`
on Sude's side) — the goal is clear boundaries between grades, not more grades.

## The scale

| Grade | Meaning |
|---|---|
| **3** | Directly answers the query; sufficient to support the answer on its own. |
| **2** | Meaningfully helps answer the query, but is incomplete or requires additional context/evidence. |
| **1** | Related to the query/topic but does not actually help answer the specific question. |
| **0** | Not relevant to answering the query. |

## What NOT to grade on

Correctness, completeness, confidence, citation quality, faithfulness,
overstated certainty, writing quality. Each of those is a real thing to
measure — they just aren't measured here. Scoring them into `grade` would
contaminate the one purpose this dataset has: retrieval quality, decoupled
from everything downstream of it.

```
retrieval → 0/1/2/3 (this file) → qrels → nDCG/MRR → DPO reranker
                                                          ↓
                                              retrieved evidence → LLM answer
                                                          ↓
                                    Sude's answer/faithfulness verdicts (separate axis)
```

If a chunk is perfectly relevant but the paper's claim turns out to be wrong,
or the eventual draft misquotes it — that's not this dataset's problem. Grade
the chunk as if it were about to be handed to someone who will read it
correctly and cite it correctly.

## Worked example

Query (`fr001`): *"What techniques are introduced to alleviate the
inconsistency problem when applying graph neural networks to fraud
detection?"* — target paper: `b8f6924f77`.

**Grade 3** — `135d84fa8d98d2eb`, section *"4.2 The Inconsistency Problem"*:
> "We first take the Yelpchi dataset to demonstrate the inconsistency
> problem..." — this is the paper's own technical section naming and
> explaining the problem and its handling. A reader gets the actual technique
> from this chunk alone.

**Grade 2** — `2d145e5b2de10098`, section *"5 CONCLUSION AND FUTURE WORKS"*:
> "In this paper, we investigate three inconsistency problems in
> apply[ing]..." — confirms the paper addresses the question and names that
> there are three problems, but doesn't itself state what the techniques
> *are*. Useful context, not sufficient alone.

**Grade 1** — `b5178028752c4f84`, the paper's `ABSTRACT`/author-list chunk:
right paper, right topic, but this particular chunk is boilerplate (author
affiliations) that doesn't touch the technique at all.

**Grade 0** — `8633f60df35a7206`, from an unrelated paper (`84ccbefe24`,
credit-card fraud via SMOTE + neural nets): different problem (class
imbalance, not GNN inconsistency), different method entirely. Retrieved only
because both papers mention "fraud detection."

## Workflow

1. `python scripts/build_eval_drafts_B.py --view fr001-fr010` — prints pooled
   candidates per query with the two most query-relevant sentences already
   picked out.
2. For each chunk you want to grade 2 or 3, add it to
   `eval/datasets/drafts/llm_draft_grades_B.json`:
   `{"fr001": {"<chunk_id>": 3, "<chunk_id>": 2}}`. Everything you don't list
   defaults to 1 (unread chunk from the target paper) or 0 (chunk from a
   non-target paper) — you only need to override the interesting cases.
3. Re-run `python scripts/build_eval_drafts_B.py` (no `--view`) to regenerate
   `qrels_draft_B.jsonl` from your judgments. Check the printed per-query
   table for `NO_GRADE3` (nothing graded a 3 yet — either genuinely
   unanswerable, or not graded yet) and `BROAD` (implausibly many grade-3s for
   one query).
4. A 50-item subset should be labeled independently by a second person before
   these drafts become the frozen `queries_B.jsonl`/`qrels_B.jsonl` — one
   annotator's judgment isn't enough to know the labels are reliable, which is
   what Sude's calibration work is validated against.
