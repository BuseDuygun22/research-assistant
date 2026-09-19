# Eval set gaps — proposed additions for review

Written 2026-09-19. These are proposals, not committed additions — each
unanswerable-query candidate below was checked against the real BM25 index
over the actual 30-paper corpus (not guessed), but "checked with BM25" is not
the same as "certified unanswerable by someone who read every paper." Review
before adding to `eval/datasets/drafts/query_specs_B.json`.

## Unanswerable-query candidates (target: ~10-15% of the eval set)

None of these terms appear in any retrieved chunk's actual text — only
generic "fraud detection" vocabulary overlap pulled candidates back at all.

| Candidate query | Checked against | Why it's unanswerable here |
|---|---|---|
| "How does differential privacy get applied to fraud detection models?" | `differential privacy` | No chunk mentions differential privacy; corpus has no privacy-preserving-ML papers |
| "How do federated learning approaches handle cross-bank fraud detection without sharing raw data?" | `federated learning`, `federated` | No chunk mentions federated learning |
| "What regulatory penalties has the SEC imposed on companies for financial fraud?" | `SEC`, `securities and exchange`, `regulatory penalt` | Corpus is purely technical/ML; no legal or regulatory-enforcement content |
| "What was Enron's accounting fraud scheme and how could ML have detected it?" | `enron` | No mention anywhere in the corpus |

Two more I checked and **rejected** as not actually unanswerable (kept here so
the same ground isn't re-covered by mistake):
- "GDPR compliance for fraud detection data" — `410803359a` gestures at
  "privacy-aware explanations" in its future-work section. Borderline grade-1,
  not grade-0.
- "Real-time streaming fraud detection latency" — genuinely covered by
  `68ce5f1dce` (Interleaved Sequence RNNs), which is explicitly about
  real-time transaction processing.
- "Adversarial attacks on fraud detection models" — `db99b0939e`'s
  Future-Directions section has one substantive paragraph naming this as an
  open problem. Grade-1 (topic named, not explained), not grade-0.

**Still needed:** ~15-20 more unanswerable candidates to reach a 10-15% share
of a 200-query set. These four are a start, not the requirement.

## Conflict queries (E38 — freshness / disagreement / incompatible scope)

**Not proposed here.** Verifying two papers genuinely disagree (rather than
simply covering different scopes, which is the normal case in a 30-paper
corpus spanning 2016-2026) needs closer reading than a keyword check can
confirm — the risk of proposing a "conflict" that turns out to be two papers
just answering different questions is real, and a wrong example would be
worse than no example. This needs a human pass (or a deeper LLM-assisted read
per paper pair) rather than a BM25 spot-check.

One structural lead worth following, not a verified answer: `b8f6924f77`
(2020) argues that applying GNN directly to fraud detection has an
"inconsistency problem" requiring specific handling. Later GNN-fraud papers in
the corpus that don't reference this problem could be read either as
genuinely disagreeing that it matters, or as simply not addressing it —
distinguishing those two readings is exactly the work that still needs doing.
