"""Model registry for rerankers — Track A / Buse, promoted from notebook 09.

Why a registry exists at all
----------------------------
Without it, "use the tuned reranker instead of the baseline" is a code change in
`retrieval/service_B.py`: an import swap, a review, a deploy. With it, retrieval
asks for a ranker *by name* (or for whatever the champion currently is) and the
registry resolves that name to either the naive ranker or a trained cross-encoder
directory. Nothing upstream — not the retrieval service, not Sude's MCP tools —
knows which model answered. That indirection is what makes the promotion gate and
the CI/CD story in the report real rather than aspirational.

Three properties are deliberate:

* **Entries are append-only.** `register()` refuses to overwrite an existing name.
  Rollback must be editing one line (`"champion"`), never retraining a model that
  was silently clobbered. A new run gets a new name.
* **Promotion is separate from registration.** Training registers; only the gate
  promotes. `register()` never touches the champion.
* **The manifest is a JSON file in the repo**, not MLflow's registry or a database,
  because Sude's gate must read it in CI with no credentials and no running
  service. MLflow still owns the runs and the metrics; this file only names the
  champion and points at artefacts.

Import cost
-----------
This module is imported by `retrieval/service_B.py`, which runs under the *base*
install. torch/transformers are therefore imported lazily, inside
`CrossEncoderRanker.rank`/`_load`, and never at import time. Importing this module
with only the base dependencies installed works and yields the baseline ranker.
Asking for a cross-encoder entry without the heavy extra raises a clear,
actionable error instead of an ImportError traceback from three frames down.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from research_assistant import config_J
from research_assistant.contracts.retrieval_J import Chunk, RetrievedChunk
from research_assistant.reranker.baseline_B import BaselineRanker, RankedChunk, Ranker

__all__ = [
    "UnknownRankerError",
    "DuplicateRankerError",
    "RerankerDependencyError",
    "CrossEncoderRanker",
    "get_ranker",
    "register",
    "promote",
    "list_rankers",
    "load_registry",
    "save_registry",
    "registry_path",
    "load_reranker_config",
    "repo_root",
    "resolve_path",
    "DEFAULT_REGISTRY",
]

BASELINE_NAME = "baseline"

#: Seed manifest used when `models/registry_B.json` does not exist yet.
DEFAULT_REGISTRY: dict[str, Any] = {
    "champion": BASELINE_NAME,
    "entries": {
        BASELINE_NAME: {
            "kind": "similarity",
            "path": None,
            "base_model": None,
            "notes": "fusion order from stage 04, no cross-encoder",
        }
    },
}


class UnknownRankerError(KeyError):
    """Asked for a ranker name that is not in the manifest."""


class DuplicateRankerError(ValueError):
    """Tried to register a name that already exists. Entries are append-only."""


class RerankerDependencyError(RuntimeError):
    """A trained cross-encoder was requested but the heavy `train` extra is absent."""


# --------------------------------------------------------------------------- #
# config access
#
# Everything here goes through the joint loader (`config_J`); nothing in this
# package opens a YAML by hand except when a CLI is pointed at a non-default
# config file, which the Makefile targets do not do but an ablation sweep will.
# --------------------------------------------------------------------------- #
def repo_root() -> Path:
    """Absolute path of the repository root, as the joint config loader sees it."""
    return config_J.REPO_ROOT


def resolve_path(rel: str | Path) -> Path:
    """Turn a repo-relative config path into an absolute one."""
    return config_J.resolve_path(rel)


def load_reranker_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load `configs/reranker_B.yaml`, or an explicit override path (the CLIs pass one)."""
    if path is not None:
        explicit = resolve_path(path)
        if explicit != resolve_path("configs/reranker_B.yaml"):
            with explicit.open(encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
    return config_J.load_config("reranker")


def registry_path(config: dict[str, Any] | None = None) -> Path:
    """Absolute path of the registry manifest named by `registry.manifest`."""
    cfg = config if config is not None else load_reranker_config()
    manifest = cfg.get("registry", {}).get("manifest", "models/registry_B.json")
    return resolve_path(manifest)


# --------------------------------------------------------------------------- #
# manifest persistence
# --------------------------------------------------------------------------- #
def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    """Read the manifest, returning the seed manifest when the file is absent."""
    manifest = Path(path) if path is not None else registry_path()
    if not manifest.exists():
        return json.loads(json.dumps(DEFAULT_REGISTRY))
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data.setdefault("entries", {})
    data["entries"].setdefault(BASELINE_NAME, DEFAULT_REGISTRY["entries"][BASELINE_NAME])
    data.setdefault("champion", BASELINE_NAME)
    return data


def save_registry(registry: dict[str, Any], path: str | Path | None = None) -> Path:
    """Write the manifest back, pretty-printed so a diff is reviewable."""
    manifest = Path(path) if path is not None else registry_path()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(registry, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return manifest


def list_rankers(path: str | Path | None = None) -> list[str]:
    """Every registered name, champion first is *not* implied — this is sorted."""
    return sorted(load_registry(path)["entries"])


def register(
    name: str,
    *,
    path: str | Path | None,
    base_model: str | None = None,
    mlflow_run_id: str | None = None,
    metrics: dict[str, float] | None = None,
    notes: str = "",
    kind: str = "cross_encoder",
    registry_file: str | Path | None = None,
) -> dict[str, Any]:
    """Add a new entry. Never overwrites: a re-run must pick a new name.

    Registration deliberately does not change the champion. Promotion is the
    gate's job.
    """
    registry = load_registry(registry_file)
    if name in registry["entries"]:
        raise DuplicateRankerError(
            f"ranker {name!r} is already registered; entries are append-only so that "
            "rollback is one line. Register the new run under a different name."
        )
    registry["entries"][name] = {
        "kind": kind,
        "path": str(path) if path is not None else None,
        "base_model": base_model,
        "mlflow_run_id": mlflow_run_id,
        "metrics": metrics or {},
        "notes": notes,
        "registered_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    save_registry(registry, registry_file)
    return registry["entries"][name]


def promote(name: str, registry_file: str | Path | None = None) -> dict[str, Any]:
    """Point the champion at an existing entry. Only after the gate passes."""
    registry = load_registry(registry_file)
    if name not in registry["entries"]:
        raise UnknownRankerError(
            f"cannot promote unknown ranker {name!r}; registered: {sorted(registry['entries'])}"
        )
    registry["champion"] = name
    save_registry(registry, registry_file)
    return registry


# --------------------------------------------------------------------------- #
# trained ranker
# --------------------------------------------------------------------------- #
class CrossEncoderRanker:
    """A tuned cross-encoder, loaded lazily so the base install can import this module."""

    kind = "cross_encoder"

    def __init__(
        self,
        name: str,
        model_path: str | Path,
        *,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self.name = name
        # `contracts.retrieval_J.Reranker` wants `.version`; the registry name
        # already carries the versioning discipline ("bump on any change that
        # would make old and new scores incomparable" -- same convention as
        # corpus_version), so it is reused rather than tracked twice.
        self.version = name
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.batch_size = batch_size
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CrossEncoderRanker(name={self.name!r}, model_path={str(self.model_path)!r})"

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            try:
                import torch  # noqa: F401  (imported for its side effect of being present)
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise RerankerDependencyError(
                    f"ranker {self.name!r} is a trained cross-encoder and needs the heavy "
                    'extra. Install it with `pip install -e ".[train]"` (or `make '
                    "install-train`), or select the baseline ranker via "
                    "`rerank.active: baseline` in configs/retrieval_B.yaml."
                ) from exc
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"registry entry {self.name!r} points at {self.model_path}, which does not "
                    "exist. Re-run training or roll the champion back to 'baseline'."
                )
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            self._model.eval()
        return self._model, self._tokenizer

    def rank(self, query: str, chunks: Sequence[Chunk]) -> list[RankedChunk]:
        """Score every (query, chunk) jointly and sort. Raises if torch is missing."""
        if not chunks:
            return []
        model, tokenizer = self._load()
        import torch

        scored: list[RankedChunk] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            encoded = tokenizer(
                [query] * len(batch),
                [chunk.text for chunk in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**encoded).logits
            scored.extend(
                (chunk, float(logit[0])) for chunk, logit in zip(batch, logits, strict=True)
            )
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """`contracts.retrieval_J.Reranker` conformance. See
        `BaselineRanker.rerank` for why this and `.rank()` are two methods."""
        chunks = [c.chunk for c in candidates]
        ranked = self.rank(query, chunks)
        by_chunk_id = {c.chunk.chunk_id: c for c in candidates}
        results: list[RetrievedChunk] = []
        for position, (chunk, score) in enumerate(ranked[:top_k], start=1):
            origin = by_chunk_id.get(chunk.chunk_id)
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(score),
                    bm25_score=origin.bm25_score if origin else None,
                    vector_score=origin.vector_score if origin else None,
                    rerank_score=float(score),
                    rank=position,
                )
            )
        return results


# --------------------------------------------------------------------------- #
# the function retrieval depends on
# --------------------------------------------------------------------------- #
def get_ranker(
    name: str | None = None,
    *,
    registry_file: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Ranker:
    """Resolve a ranker name to an object with `.name` and `rank(query, chunks)`.

    `name=None` resolves to the current champion, which is what
    `retrieval/service_B.py` should pass when `rerank.active` is unset. An unknown
    name is an error, never a silent fallback to the baseline: a typo in a config
    must fail loudly rather than quietly ship the wrong model and make every
    recorded number unattributable.
    """
    registry = load_registry(registry_file)
    resolved = name if name is not None else registry.get("champion", BASELINE_NAME)
    entries = registry["entries"]
    if resolved not in entries:
        raise UnknownRankerError(
            f"unknown ranker {resolved!r}; registered: {sorted(entries)}. "
            "Check `rerank.active` in configs/retrieval_B.yaml against models/registry_B.json."
        )
    entry = entries[resolved]
    kind = entry.get("kind", "similarity")
    if kind in {"similarity", "baseline"}:
        return BaselineRanker(
            name=resolved,
            mode=entry.get("mode", "fusion_order"),
            embedding_model=entry.get("embedding_model"),
        )
    if kind == "cross_encoder":
        cfg = config if config is not None else load_reranker_config()
        model_path = entry.get("path")
        if not model_path:
            raise UnknownRankerError(
                f"registry entry {resolved!r} has kind 'cross_encoder' but no path"
            )
        return CrossEncoderRanker(
            name=resolved,
            model_path=resolve_path(model_path),
            max_length=int(cfg.get("model", {}).get("max_length", 512)),
        )
    raise UnknownRankerError(f"registry entry {resolved!r} has unsupported kind {kind!r}")
