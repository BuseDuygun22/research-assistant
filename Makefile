# Shared Makefile. Track A targets below are Buse's; the Track B section is Sude's.
# On Windows run these through Git Bash, or copy the command out of the recipe.

PY := .venv/Scripts/python

.PHONY: help install install-train install-obs kernel lint test \
        ingest pairs train eval gate clean-index

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

# --- setup -------------------------------------------------------------------
install:        ## base deps + notebooks + dev tooling
	$(PY) -m pip install -e ".[notebooks,dev,obs]"

install-train:  ## heavy ML deps, needed from stage 08
	$(PY) -m pip install -e ".[train]"

kernel:         ## register the Jupyter kernel used by the notebooks
	$(PY) -m ipykernel install --user --name research-assistant \
		--display-name "Python 3 (research-assistant)"

lint:
	$(PY) -m ruff check src eval scripts
	$(PY) -m ruff format --check src eval scripts

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
	$(PY) -m mlflow ui --backend-store-uri ./mlruns

clean-index:    ## drop the local vector and sparse indexes, keeps data/raw
	rm -rf data/chroma data/bm25_index_B.pkl data/processed data/interim

# --- Track B (Sude) ----------------------------------------------------------
# serve, docker-build, gate: owned by Sude, see her track.
gate:           ## run the promotion gate locally (Sude's runner, Buse's thresholds)
	$(PY) eval/run_gate_S.py --thresholds eval/thresholds_B.yaml
