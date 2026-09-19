"""DPO cross-encoder training — Track A / Buse, promoted from notebook 08.

What this trains and why
------------------------
The stage 04 candidate set is ordered by fusion rank, which knows only term
overlap and embedding proximity. A cross-encoder reads the query and the chunk
*together*, so it can judge relevance rather than similarity. That is why it can
win, and why it only runs over ~20 candidates instead of the corpus.

The objective is DPO because the data is *preferences* (stage 07 triples), not
calibrated scores: taking the judge's 0-3 grades as regression targets would
train the model to reproduce the judge's scale, including its noise, whereas DPO
only asks that chosen beat rejected. `beta` keeps the tuned model from drifting
far from the reference model, which is the right default on a few hundred pairs.
One epoch, for the same reason: more epochs memorise the judge.

Expect the first run to lose to the baseline. That is a legitimate result, it
belongs in the report, and the registry plus gate exist to stop it shipping.

Why this module imports without torch
-------------------------------------
`torch`, `transformers`, `datasets` and `trl` come from the optional `train` extra,
which is not installed in the working venv (`torch`/`transformers` happen to arrive
transitively with `sentence-transformers`, but `datasets` and `trl` do not, so
training is unavailable either way). Everything that can be checked without
them — config parsing, dataset construction, MLflow parameter assembly — is a
plain function at module level with no heavy import. The heavy imports happen
inside `train()`, behind `require_training_deps()`, which raises a single
actionable `TrainingDependenciesMissing` instead of an ImportError from deep in a
library. So `import research_assistant.reranker.train_B` is safe everywhere
(including in Sude's CI, which installs only the base deps), and only actually
invoking training demands the extra.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_assistant.contracts.judge_J import PreferencePair
from research_assistant.reranker.registry_B import load_reranker_config, resolve_path

__all__ = [
    "TrainingDependenciesMissing",
    "TrainSettings",
    "REQUIRED_PACKAGES",
    "missing_dependencies",
    "require_training_deps",
    "load_pairs",
    "build_dpo_rows",
    "split_rows",
    "training_params",
    "log_to_mlflow",
    "train",
]

#: Distribution imports that the `train` extra provides.
REQUIRED_PACKAGES: tuple[str, ...] = ("torch", "transformers", "datasets", "trl")


class TrainingDependenciesMissing(RuntimeError):
    """Training was invoked without the heavy `train` extra installed."""


@dataclass(frozen=True)
class TrainSettings:
    """The subset of `configs/reranker_B.yaml` that training actually uses.

    Parsed as a dataclass so a typo in the YAML fails here, with a readable name,
    rather than four minutes into a run.
    """

    base_model: str
    max_length: int
    num_labels: int
    beta: float
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    warmup_ratio: float
    max_grad_norm: float
    seed: int
    output_dir: Path
    pairs_path: Path
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str | None = None
    log_params: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TrainSettings:
        model = config["model"]
        train = config["train"]
        pairs = config["pairs"]
        mlflow_cfg = config.get("mlflow", {})
        method = str(train.get("method", "dpo")).lower()
        if method != "dpo":
            raise ValueError(
                f"train.method is {method!r}; this module implements DPO only. "
                "A pairwise-hinge ablation belongs in its own module and ledger row."
            )
        return cls(
            base_model=str(model["base"]),
            max_length=int(model.get("max_length", 512)),
            num_labels=int(model.get("num_labels", 1)),
            beta=float(train["beta"]),
            learning_rate=float(train["learning_rate"]),
            per_device_train_batch_size=int(train.get("per_device_train_batch_size", 8)),
            gradient_accumulation_steps=int(train.get("gradient_accumulation_steps", 1)),
            num_train_epochs=float(train.get("num_train_epochs", 1)),
            warmup_ratio=float(train.get("warmup_ratio", 0.1)),
            max_grad_norm=float(train.get("max_grad_norm", 1.0)),
            seed=int(train.get("seed", 42)),
            output_dir=resolve_path(train["output_dir"]),
            pairs_path=resolve_path(pairs["out_path"]),
            mlflow_tracking_uri=mlflow_cfg.get("tracking_uri"),
            mlflow_experiment=mlflow_cfg.get("experiment"),
            log_params=bool(mlflow_cfg.get("log_params", True)),
            extra={"min_score_gap": pairs.get("min_score_gap")},
        )


# --------------------------------------------------------------------------- #
# dependency gate
# --------------------------------------------------------------------------- #
def missing_dependencies() -> list[str]:
    """Names from the `train` extra that are not importable. Does not import them."""
    return [pkg for pkg in REQUIRED_PACKAGES if importlib.util.find_spec(pkg) is None]


def require_training_deps() -> None:
    """Raise a single actionable error when the heavy extra is absent."""
    missing = missing_dependencies()
    if missing:
        raise TrainingDependenciesMissing(
            "DPO training needs the optional 'train' extra, which is not installed. "
            f"Missing: {', '.join(missing)}. Install it with "
            '`.venv/Scripts/python -m pip install -e ".[train]"` (or `make install-train`) '
            "and re-run `make train`. Building pairs (`make pairs`), the registry and the "
            "baseline ranker all work without it."
        )


# --------------------------------------------------------------------------- #
# torch-free pieces: data and logging
# --------------------------------------------------------------------------- #
def load_pairs(path: str | Path) -> list[PreferencePair]:
    """Read the stage 07 JSONL back into `PreferencePair` objects."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"preference pairs not found at {file}. Run `make pairs` (stage 07) first."
        )
    pairs: list[PreferencePair] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pairs.append(PreferencePair.model_validate_json(line))
    return pairs


def build_dpo_rows(pairs: Sequence[PreferencePair]) -> list[dict[str, str]]:
    """Convert triples into the prompt/chosen/rejected rows `trl` expects.

    Plain dicts, not a `datasets.Dataset`, so this is testable under the base
    install; `train()` wraps the result in a Dataset once the extra is present.
    """
    if not pairs:
        raise ValueError("no preference pairs to train on; stage 07 produced an empty file")
    return [
        {"prompt": pair.query, "chosen": pair.chosen_text, "rejected": pair.rejected_text}
        for pair in pairs
    ]


def split_rows(
    rows: Sequence[dict[str, str]], *, test_size: float = 0.1, seed: int = 42
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Deterministic train/held-out split of the DPO rows.

    This split is only a training-time sanity signal. The evaluation that decides
    promotion is the stage 05 held-out query set, run through the same code path
    as the baseline.
    """
    import random

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_size)) if len(shuffled) > 1 else 0
    return shuffled[n_test:], shuffled[:n_test]


def training_params(settings: TrainSettings, n_pairs: int) -> dict[str, Any]:
    """The parameter dict logged to MLflow. Pure, so a test can assert on it."""
    return {
        "base_model": settings.base_model,
        "beta": settings.beta,
        "lr": settings.learning_rate,
        "epochs": settings.num_train_epochs,
        "batch_size": settings.per_device_train_batch_size,
        "grad_accum": settings.gradient_accumulation_steps,
        "max_length": settings.max_length,
        "seed": settings.seed,
        "n_pairs": n_pairs,
        "min_score_gap": settings.extra.get("min_score_gap"),
    }


def log_to_mlflow(settings: TrainSettings, params: dict[str, Any], *, run_name: str = "dpo"):
    """Open an MLflow run with the params already logged; returns the run context.

    mlflow is a *base* dependency, so this path is exercised without the heavy
    extra. Used as a context manager by `train()`.
    """
    import mlflow

    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    if settings.mlflow_experiment:
        mlflow.set_experiment(settings.mlflow_experiment)
    run = mlflow.start_run(run_name=run_name)
    if settings.log_params:
        mlflow.log_params(params)
    return run


# --------------------------------------------------------------------------- #
# the gated training path
# --------------------------------------------------------------------------- #
def train(
    config: dict[str, Any] | None = None,
    *,
    config_path: str | Path | None = None,
    run_name: str = "dpo_v1",
    register_as: str | None = None,
) -> dict[str, Any]:
    """Fine-tune the cross-encoder with DPO. Requires the `train` extra.

    Returns a summary dict (output dir, MLflow run id, number of pairs). It
    deliberately does **not** promote anything: registration is optional here and
    promotion belongs to the gate.
    """
    cfg = config if config is not None else load_reranker_config(config_path)
    settings = TrainSettings.from_config(cfg)
    pairs = load_pairs(settings.pairs_path)
    rows = build_dpo_rows(pairs)

    # Everything above this line runs under the base install. Everything below
    # needs the heavy extra, so the gate sits exactly here.
    require_training_deps()

    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    train_rows, eval_rows = split_rows(rows, seed=settings.seed)
    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows) if eval_rows else None

    tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.base_model, num_labels=settings.num_labels
    )
    ref_model = AutoModelForSequenceClassification.from_pretrained(
        settings.base_model, num_labels=settings.num_labels
    )

    args = DPOConfig(
        output_dir=str(settings.output_dir),
        beta=settings.beta,
        learning_rate=settings.learning_rate,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        num_train_epochs=settings.num_train_epochs,
        warmup_ratio=settings.warmup_ratio,
        max_grad_norm=settings.max_grad_norm,
        seed=settings.seed,
        max_length=settings.max_length,
        logging_steps=10,
        report_to=[],
    )

    params = training_params(settings, len(pairs))
    with log_to_mlflow(settings, params, run_name=run_name) as run:
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
        )
        trainer.train()
        trainer.save_model(str(settings.output_dir))
        tokenizer.save_pretrained(str(settings.output_dir))
        run_id = run.info.run_id

    summary = {
        "output_dir": str(settings.output_dir),
        "mlflow_run_id": run_id,
        "n_pairs": len(pairs),
        "base_model": settings.base_model,
    }

    if register_as:
        from research_assistant.reranker.registry_B import register

        register(
            register_as,
            path=settings.output_dir,
            base_model=settings.base_model,
            mlflow_run_id=run_id,
            notes=f"DPO, beta={settings.beta}, {len(pairs)} pairs",
        )
        summary["registered_as"] = register_as
    return summary
