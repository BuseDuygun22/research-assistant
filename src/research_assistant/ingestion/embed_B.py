"""Stage 03 — sentence-transformers wrapper.

Decides the embedding model and, more importantly, enforces the asymmetric prefix
convention. Reasoning and the model comparison table are in
`notebooks/03_embedding_vectorstore_B.ipynb`.

The one thing this module exists to prevent: `bge-*` and `e5-*` are trained with a
query-side instruction, and applying that instruction to passages -- or omitting it on
queries -- costs real accuracy while looking exactly like a mediocre model. It is the
most common silent misuse of this model family. So there is no general `embed(texts)`
here. `embed_passages` and `embed_query` are separate functions with separate prefixes
read from `configs/ingestion_B.yaml` under `embed:`, and neither can be reached by the
other's argument.

The model is lazy-loaded behind `get_model()`, so importing this module stays cheap.
`pipeline_B` imports it during a `--stage parse` run and must not pay for a 33M
parameter checkpoint to do so.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np

from research_assistant.config_J import load_config

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from sentence_transformers import SentenceTransformer


def embed_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """The `embed:` block of the ingestion config."""
    return (config or load_config("ingestion"))["embed"]


@functools.lru_cache(maxsize=2)
def get_model(model_name: str, device: str = "auto") -> SentenceTransformer:
    """Load and cache the sentence-transformers model.

    Cached on `(model_name, device)` so a process that embeds passages and then queries
    loads one copy. `device="auto"` is passed through as `None`, which is what
    sentence-transformers itself interprets as "pick CUDA if it is there".
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=None if device == "auto" else device)


def load_embedder(config: dict[str, Any] | None = None) -> SentenceTransformer:
    """Load the model named in config and assert its dimension matches the declared one.

    The assertion is not defensive noise. `embed.dim` is what the vector store's
    collection is created with, and a silent dimension mismatch surfaces much later as
    an unexplained upsert failure or, worse, a collection that accepts garbage.
    """
    cfg = embed_config(config)
    model = get_model(cfg["model"], cfg.get("device", "auto"))
    actual = model.get_sentence_embedding_dimension()
    expected = int(cfg["dim"])
    if actual != expected:
        raise ValueError(
            f"embedding dim mismatch: {cfg['model']} produces {actual}, "
            f"configs/ingestion_B.yaml declares {expected}. Update embed.dim."
        )
    return model


def embed_passages(
    texts: list[str],
    *,
    config: dict[str, Any] | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed chunk texts for indexing. Applies `passage_prefix`, never `query_prefix`."""
    if not texts:
        return np.zeros((0, int(embed_config(config)["dim"])), dtype=np.float32)
    cfg = embed_config(config)
    model = load_embedder(config)
    prefix = cfg.get("passage_prefix", "") or ""
    vectors = model.encode(
        [prefix + t for t in texts],
        batch_size=int(cfg.get("batch_size", 32)),
        normalize_embeddings=bool(cfg.get("normalize", True)),
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str, *, config: dict[str, Any] | None = None) -> np.ndarray:
    """Embed one search query. Applies `query_prefix`, which passages must never see.

    Returns a 1-D vector rather than a batch of one, because every caller downstream
    wants a single vector and the squeeze is the place mistakes happen.
    """
    cfg = embed_config(config)
    model = load_embedder(config)
    prefix = cfg.get("query_prefix", "") or ""
    vector = model.encode(
        [prefix + query],
        normalize_embeddings=bool(cfg.get("normalize", True)),
        convert_to_numpy=True,
    )[0]
    return np.asarray(vector, dtype=np.float32)


def embed_queries(queries: list[str], *, config: dict[str, Any] | None = None) -> np.ndarray:
    """Batch form of `embed_query`, for evaluation runs over a whole query set."""
    if not queries:
        return np.zeros((0, int(embed_config(config)["dim"])), dtype=np.float32)
    cfg = embed_config(config)
    model = load_embedder(config)
    prefix = cfg.get("query_prefix", "") or ""
    vectors = model.encode(
        [prefix + q for q in queries],
        batch_size=int(cfg.get("batch_size", 32)),
        normalize_embeddings=bool(cfg.get("normalize", True)),
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)
