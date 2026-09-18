# Editorial rubric — what makes a draft publishable

Owner: **Sude**. This document is executable: `judge/faithfulness_S.py` derives
`FaithfulnessVerdict.passed` from the rules below, and the editor node routes on
that boolean. A change here changes agent behaviour, so change it deliberately.

The rubric exists because of one decision in `architecture_J.md`: **`passed` is
derived by a rule, not self-reported by the model.** A model asked "did you pass?"
is grading its own work, and the editor routes on the answer. Counted violations
against a fixed threshold is a decision anyone can audit; a model's opinion of its
own output is not.

---

## 1. Claim decomposition

Before anything is graded, the draft is split into **atomic claims**: one
assertion each, no conjunctions, no "and therefore". A sentence carrying two
assertions becomes two claims.

Why atomic: a verifier that can only accept or reject whole sentences cannot say
"the first half is supported and the second is not", which is the most common
real failure. Decomposition is what makes `citation_precision` a measurement
rather than an impression.

**Not claims** (excluded from all counts): the question restated, transitions
("This section compares…"), and explicit hedges about the corpus itself ("the
retrieved papers do not address X"). Counting these as uncited claims would
penalise exactly the honest behaviour we want.

---

## 2. Violation kinds

Every claim that fails is labelled with exactly one kind. The **depth** column is
load-bearing: it decides which node repairs the draft.

| Kind | Depth | Test |
|---|---|---|
| `missing_citation` | shallow | A factual claim with no citation attached, where supporting evidence *is* present in the retrieved context |
| `overstated_certainty` | shallow | The source hedges ("suggests", "in our setting") and the draft asserts flatly |
| `misattributed_citation` | shallow | The claim is supported by a retrieved chunk, but cites a *different* one |
| `unsupported_claim` | **deep** | A factual claim no retrieved chunk supports |
| `inferential_leap` | **deep** | The cited span supports A; the claim asserts B, which does not follow from A alone |
| `contradicts_source` | **deep** | The cited span asserts the opposite of the claim |

**Choosing between `unsupported_claim` and `inferential_leap`:** if there is *no*
relevant span, it is unsupported. If there is a span that gets you part of the
way and the draft has jumped the rest, it is a leap. The distinction matters
because the repair differs — one needs new evidence, the other needs the claim
weakened to what the evidence actually licenses.

**Choosing between `overstated_certainty` and `inferential_leap`:** overstatement
is about *strength* ("may reduce" → "reduces"). A leap is about *content* (B is a
different proposition from A). Overstatement is repairable by hedging, which is
why it is shallow.

---

## 3. Severity

| Severity | Applies when |
|---|---|
| `major` | The claim is load-bearing: remove it and the answer changes |
| `minor` | The claim is incidental — background, an aside, a restatement |

Default is `major`. Downgrading is a positive judgement the judge must justify in
`explanation`, because the easy failure here is a judge that calls everything
minor and lets the draft through.

---

## 4. The pass rule

A draft **passes** when all of:

1. **No deep violations.** Not one. A single `unsupported_claim`,
   `inferential_leap` or `contradicts_source` fails the draft, regardless of
   everything else. These are the failures that make a cited report worse than no
   report, because the citation lends unearned credibility.
2. **No `major` shallow violations.**
3. **`citation_precision >= 0.95`** — of the claims that carry a citation, at
   least 95% are genuinely supported.
4. **`coverage >= 0.80`** — at least 80% of factual claims carry a citation. Not
   100%, because §1 excludes transitions and hedges, and a rubric that demands a
   citation on every sentence produces citation-stuffing rather than grounding.

Minor shallow violations are reported and do not block. They are still counted,
still traced, and still routed if the draft fails for another reason.

**Thresholds are `[D]` — team decisions, not findings.** 0.95 and 0.80 are set to
be measurable and revisable; once the judge is calibrated against Buse's qrels we
will know what they cost in escalation rate and can move them with evidence.

---

## 5. What this rubric does *not* check

Stated explicitly, because the gap is easy to miss and dangerous to forget.

This rubric measures **attribution** — does the draft match its sources. It does
not measure:

- **Whether the sources are right.** A draft faithfully reporting a retracted
  paper passes every rule above. No amount of grounding discipline catches this;
  it needs reference answers, which the eval set does not carry.
- **Whether the draft answers the question.** Five verbatim quotes score 1.0 on
  every rule here and answer nothing. That is `AnswerVerdict`'s job, graded
  separately and routed separately.
- **Whether the evidence was complete.** Each claim can be supported while the
  retrieved set covers one side of a disagreement. Context recall against qrels
  catches it in evaluation; nothing catches it at runtime.

See `docs/design_review_J.md` §4 (E6, E7) for the full statement of these limits.

---

## 6. Worked examples

Source span: *"On the WANDS benchmark, a tuned hybrid reaches 0.7497 nDCG against
0.6983 for BM25."*

| Draft claim | Verdict |
|---|---|
| "A tuned hybrid reached 0.7497 nDCG on WANDS [c1]." | pass |
| "A tuned hybrid reached 0.7497 nDCG on WANDS." | `missing_citation`, shallow |
| "Hybrid retrieval always outperforms BM25 [c1]." | `inferential_leap`, deep — one benchmark does not license "always" |
| "Hybrid retrieval outperformed BM25 [c2]." (c2 is a DPR chunk) | `misattributed_citation`, shallow |
| "Hybrid retrieval is the best method available [c1]." | `unsupported_claim`, deep |
| "BM25 outperformed hybrid on WANDS [c1]." | `contradicts_source`, deep |
| "Hybrid may slightly help on WANDS [c1]." | pass — under-claiming is not a violation |

Note the last row. Under-claiming is *not* penalised by this rubric, which is
precisely why `AnswerVerdict` has to exist: on attribution alone, the safest
possible draft is the one that says almost nothing.
