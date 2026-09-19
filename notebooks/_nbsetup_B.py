"""Shared notebook setup for Track A (Buse).

Kept deliberately tiny. Anything that grows beyond loading paths and YAML belongs
in `src/research_assistant/`, not here, so the notebooks stay readable as design
documents rather than turning into a second codebase.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA = REPO / "data"
CONFIGS = REPO / "configs"

_CFG_FILES = {
    "ingestion": CONFIGS / "ingestion_B.yaml",
    "retrieval": CONFIGS / "retrieval_B.yaml",
    "reranker": CONFIGS / "reranker_B.yaml",
    "thresholds": REPO / "eval" / "thresholds_B.yaml",
}


def load_cfg(name: str) -> dict:
    """Load one Track A config by short name: ingestion | retrieval | reranker | thresholds."""
    path = _CFG_FILES[name]
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel: str) -> Path:
    """Turn a repo-relative path string from a config file into an absolute Path."""
    return (REPO / rel).resolve()


def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
