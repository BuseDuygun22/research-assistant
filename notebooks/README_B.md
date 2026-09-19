# Track A notebooks — Buse

One notebook per stage. Each is a **design document that runs**: it states what the
stage decides, the choice made, the alternatives rejected and why, the prototype code,
and the checks that must pass before the next stage starts.

Notebooks are where a stage is explored and defended. `src/research_assistant/` is
where the settled version lives. Each notebook names the module it promotes to, and
nothing in a notebook is imported by production code.

## Stage map

| # | Notebook | Decides | Promotes to | Depends on |
|---|---|---|---|---|
| 00 | `00_corpus_scoping_B.ipynb` | What the assistant answers, which papers, which tactics to try | `docs/domain_brief_B.md`, `docs/tactics_ledger_B.md` | — |
| 01 | `01_parsing_B.ipynb` | What information survives the PDF | `ingestion/parse_B.py` | 00 |
| 02 | `02_chunking_B.ipynb` | The retrieval ceiling | `ingestion/chunk_B.py` | 01 |
| 03 | `03_embedding_vectorstore_B.ipynb` | Embedding model, index layout | `ingestion/embed_B.py`, `retrieval/vector_store_B.py` | 02 |
| 04 | `04_hybrid_retrieval_B.ipynb` | Sparse + dense fusion, the candidate set, the top-k contract | `retrieval/bm25_B.py`, `hybrid_B.py`, `service_B.py` | 03 |
| 05 | `05_eval_set_B.ipynb` | What counts as a relevant hit | `eval/datasets/*_B.jsonl` | 00, 02 |
| 06 | `06_metrics_baseline_B.ipynb` | The baseline and the gate thresholds | `eval/metrics/retrieval_B.py` | 04, 05 |
| 07 | `07_preference_pairs_B.ipynb` | Training signal from Sude's judge | `reranker/pairs_B.py` | 05, 06 |
| 08 | `08_dpo_reranker_B.ipynb` | Whether a tuned reranker beats the baseline | `reranker/train_B.py` | 07 |
| 09 | `09_registry_gate_B.ipynb` | How a model gets promoted or rolled back | `reranker/registry_B.py` | 08 |
| 10 | `10_observability_B.ipynb` | What every trace records | `observability/tracing_B.py` | 04 |

## Order to actually work in

The numbering follows the data. The build order does not.

1. **00 → 01 → 02 → 03 → 04.** Get a retriever that returns something.
2. **10, partially.** Instrument as soon as 04 works. Retrofitting tracing later
   across nine stages costs far more than adding it once.
3. **05 → 06.** Now you can measure, and only now do the sweeps in 02 and 04 mean
   anything. Go back and run them.
4. **06 sets the thresholds.** Hand them to Sude, her gate enforces them.
5. **07 → 08 → 09**, once Sude's judge is producing scores.

Stage 06 is the referee for every choice made in 01 through 04. Expect to loop back.

## Conventions

- **Configs are the source of truth.** A notebook reads `configs/*_B.yaml` and never
  hard-codes a value that belongs there. Change the YAML, re-run the notebook.
- **Every design-choice table is editable.** If you disagree with a pick, change it,
  re-run, and record the delta in `docs/tactics_ledger_B.md`. The tables record what was
  decided and why, not what is permanent.
- **No notebook writes to `src/`.** Promotion is a manual, deliberate copy.
- **Nothing is adopted without a number.** A tactic moves to `adopted` only after
  stage 06 measures it on the held-out set.

## Setup

```bash
.venv/Scripts/python -m ipykernel install --user --name research-assistant --display-name "Python 3 (research-assistant)"
```

Stages 00 through 07 and 09 run on the base install. Stage 08 needs the heavy extra
(`pip install -e ".[train]"`), stage 10 needs `".[obs]"`.
