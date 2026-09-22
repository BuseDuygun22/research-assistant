# Shared Makefile. Track A targets below are Buse's; the Track B section is Sude's.
# On Windows run these through Git Bash, or copy the command out of the recipe.

# Override for Linux/macOS/CI:  make PY=python <target>
PY ?= .venv/Scripts/python
Q ?= Which methods are used to handle class imbalance in fraud detection?

.PHONY: help install install-train install-obs kernel lint test \
        ingest pairs train eval gate calibrate-judge clean-index install-all demo-corpus demo ask

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

# --- setup -------------------------------------------------------------------
install:        ## base deps + notebooks + dev tooling
	$(PY) -m pip install -e ".[notebooks,dev,obs]"

install-all:    ## everything the pipeline, agents, tests and CI need (no training deps)
	$(PY) -m pip install -e ".[serve,agents,judge,eval,dev,obs]"

install-train:  ## heavy ML deps, needed from stage 08
	$(PY) -m pip install -e ".[train]"

kernel:         ## register the Jupyter kernel used by the notebooks
	$(PY) -m ipykernel install --user --name research-assistant \
		--display-name "Python 3 (research-assistant)"

lint:
	$(PY) -m ruff check src eval scripts

test:
	$(PY) -m pytest -q

# --- Track A (Buse) ----------------------------------------------------------
ingest:         ## parse, chunk, embed, index the corpus (stages 01-03)
	$(PY) scripts/ingest_B.py --config configs/ingestion_B.yaml

pairs:          ## build DPO preference pairs from the judge scores (stage 07)
	$(PY) scripts/build_pairs_B.py --config configs/reranker_B.yaml

train:          ## fine-tune the cross-encoder reranker (stage 08)
	$(PY) scripts/train_reranker_B.py --config configs/reranker_B.yaml

eval:           ## retrieval metrics on the held-out set (stage 06)
	$(PY) -m eval.metrics.retrieval_B --config configs/retrieval_B.yaml

mlflow:         ## browse training and eval runs
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

clean-index:    ## drop the local vector and sparse indexes, keeps data/raw
	rm -rf data/chroma data/bm25_index_B.pkl data/processed data/interim

# --- Track B (Sude) ----------------------------------------------------------
# serve, docker-build, gate: owned by Sude, see her track.
gate:           ## run the promotion gate locally (Sude's runner, Buse's thresholds)
	$(PY) -m eval.run_gate_S --thresholds eval/thresholds_B.yaml

calibrate-judge: ## kappa + reliability of the judge against Buse's human qrels; exit 1 below kappa 0.5
	$(PY) scripts/calibrate_judge_S.py --qrels eval/datasets/qrels_B.jsonl --queries eval/datasets/queries_B.jsonl

# --- end to end ----------------------------------------------------------------
demo-corpus:    ## seven synthetic PDFs into data/raw, so the pipeline runs with no download
	mkdir -p data/raw
	cp tests/fixtures/synthetic_pdfs_B/*.pdf data/raw/
	cp tests/fixtures/synthetic_manifest_B.jsonl data/corpus_manifest_B.jsonl

demo: demo-corpus  ## parse, chunk, embed and index the demo corpus, then answer one question
	$(PY) scripts/ingest_B.py --stage all
	RA_RETRIEVAL_BACKEND=track_a $(PY) -m research_assistant.ask_S --mlflow "$(Q)"

ask:            ## ask one question (Q="...", LLM=ollama MODEL=qwen2.5:7b-instruct)
	$(PY) -m research_assistant.ask_S $(if $(LLM),--llm $(LLM)) $(if $(MODEL),--model $(MODEL)) "$(Q)"
