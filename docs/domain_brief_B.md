# Domain brief - Track A (Buse)

Decision record for what the research assistant answers from this corpus, and what
it deliberately does not. Stage 05 samples eval queries from the question shapes
below. A retrieval miss on an out-of-scope question is not a bug.

## Scope

**In:** Machine-learning methods for detecting fraud in financial and commercial
transaction data: credit card fraud, online review and e-commerce fraud, customs
and occupational fraud. Graph neural networks, anomaly detection, boosting,
sequence models, and their explainability, as published on arXiv from 2016 to 2026.

**Excluded:** The legal, regulatory, criminological and forensic-accounting sides of
fraud, and the operational engineering of deployed fraud systems (streaming
infrastructure, case-management workflow, vendor products). The corpus contains
methods papers, not practice or policy papers, so it cannot answer those questions
with evidence.

## Question shapes the assistant must answer

Each shape is backed by at least three corpus papers. Counts are papers whose text
mentions the shape's key terms, measured over the parsed chunks on 2026-09-17.

1. **Method lookup.** "What approach does paper X propose, and what is its core idea?"
   Every paper states a method, 30/30.
2. **Dataset and setup lookup.** "Which datasets does paper X evaluate on?" Backed by
   the recurring benchmarks: YelpChi and Amazon reviews (6 papers each), the European
   credit card dataset (6), PaySim (3).
3. **Reported results and metrics.** "Which metrics does paper X report, and what score
   does it reach?" AUC appears in 30/30, recall or precision in 23, F1 in 14.
4. **Challenge handling.** "How does paper X deal with class imbalance, camouflaged
   fraudsters, or scarce labels?" Imbalance in 20 papers, label scarcity or
   semi-supervision in 13, camouflage in 9, SMOTE in 6.
5. **Explainability and stated limitations.** "How does paper X explain its decisions?"
   and "What limitations do the authors state?" Explainability or interpretability in
   19 papers, an explicit limitations discussion in 17.

## Shape that needs two papers

6. **Comparison.** "How do the GNN approaches in paper A and paper B differ in handling
   heterophily or camouflage?" This is multi-hop: the answer needs one chunk from each
   paper. GNN work appears in 14 papers, so the shape is well supported, but the stage 04
   candidate set is tuned for single-hop questions. Keep these to at most 20 percent of
   the stage 05 query set, and report their metrics separately in stage 06.

## Explicitly out of scope

- **Concept drift and temporal model decay.** Only 2 papers treat it, below the
  three-paper floor. Answers would rest on a single source.
- **Exact numbers from results tables.** PyMuPDF flattens tables into loose tokens, so
  a question like "what is the F1 of baseline Y in row 4 of Table 3" tests the parser,
  not retrieval. Questions about scores stated in running text stay in scope.
- **Cross-paper leaderboards.** "Which method is best on YelpChi?" Papers use different
  splits and metric variants, so the corpus cannot answer this honestly.
- **Anything about specific datasets beyond what papers say.** Dataset licensing,
  collection details, or access are not in these papers.
- **Questions answerable only from references.** Reference sections are dropped at
  stage 01.

## Why this corpus fits

The 30 papers were fetched from arXiv with the exact phrase "fraud detection",
restricted to machine-learning, security and quantitative-finance categories. That
restriction keeps unrelated senses of "fraud" out. The set is deliberately mixed:
three surveys give broad framing, while the other 27 are method papers that state a
method, datasets, metrics and results. That mix supports both lookup questions and
comparison questions. Recurring benchmarks, especially YelpChi, Amazon and the European
credit card data, mean several papers talk about the same things. That overlap is what
makes ranking non-trivial: a query about YelpChi has several plausible chunks, and the
retriever has to find the right one.

## Corpus bounds

- Size: 30 papers. Below the stage 00 recommendation of 40 to 80, chosen for labelling
  cost. Revisit if stage 06 metrics saturate near 1.0, which would mean the corpus is
  too easy to separate retrieval tactics.
- Years: 2016 to 2026.
- Source: arXiv, primary categories cs.LG (18), cs.CR (4), cs.SI (2), cs.AI (2),
  q-fin.ST, stat.ML, quant-ph, cs.CE (1 each).
- Manifest: `data/corpus_manifest_B.jsonl`. Fetch script: `scripts/fetch_arxiv_fraud_B.py`.

## Known corpus weaknesses

- **Parsing is uneven on math-heavy GNN papers.** After two heading-detector fixes, the
  median is 15 detected sections per paper, but the worst paper still shows 93. Section
  labels on those papers are less trustworthy for citations.
- **Three surveys overlap heavily with method papers.** A survey chunk summarising
  paper X will compete with paper X's own chunk. Stage 05 labels should grade the
  original paper's chunk higher than the survey's summary of it.

## Consequences for later stages

- Stage 05 draws queries across shapes 1 to 6, roughly evenly for 1 to 5, capped at 20
  percent for shape 6. At 30 papers, target 40 to 60 queries.
- Stage 05 grade 3 means the chunk comes from the paper the question is about. A survey
  restating the same fact is grade 2 at most.
- Stage 06 reports single-hop and comparison queries separately.
- Citation precision is measured against page ranges of these 30 papers only.
- Sude's editorial rubric should treat out-of-scope questions as correct refusals, not
  faithfulness failures.
