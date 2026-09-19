"""Ingestion orchestration: parse -> chunk -> embed -> upsert.

Decides the *order and the resumability* of stages 01 to 03, not their content. Each
stage's reasoning lives in its own module and its own notebook (01 parsing, 02
chunking, 03 embedding). What is decided here:

- **Every stage persists its intermediate.** `data/interim/pages.jsonl` and
  `data/processed/chunks.jsonl` are written even on an `all` run. Parsing is the slow
  step and chunking is the one that gets swept; keeping both artefacts means a chunking
  experiment never re-parses, which is the whole argument for a page-level intermediate
  in `notebooks/01_parsing_B.ipynb`.
- **Stages run as a prefix, not a selection.** `--stage chunk` runs parse then chunk,
  because chunking without fresh pages is how a sweep ends up measuring a stale corpus.
- **Embedding is imported lazily.** A `--stage parse` run must not load a 33M parameter
  checkpoint, so `embed_B` is imported inside the function that needs it.

The vector store is reached only through `retrieval.vector_store_B.VectorStore`
(`upsert` / `search` / `count` / `get`). This module never imports chromadb.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_assistant.config_J import load_config, resolve_path
from research_assistant.contracts.retrieval_J import Chunk
from research_assistant.ingestion.chunk_B import (
    chunk_pages,
    count_tokens,
    get_encoder,
    write_chunks,
)
from research_assistant.ingestion.parse_B import PageRecord, parse_corpus, write_pages

STAGES: tuple[str, ...] = ("parse", "chunk", "embed", "all")

#: `all` is `embed` plus the upsert. Everything else is a prefix of the stage list.
_STAGE_ORDER: dict[str, int] = {"parse": 1, "chunk": 2, "embed": 3, "all": 4}


@dataclass
class IngestionResult:
    """Everything a run produced, so the CLI can print a summary without re-deriving it."""

    stage: str
    dry_run: bool
    pages: list[PageRecord] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    parse_report: dict[str, Any] = field(default_factory=dict)
    token_stats: dict[str, float] = field(default_factory=dict)
    artefacts: list[Path] = field(default_factory=list)
    embedded: int = 0
    embedding_dim: int | None = None
    store_count: int | None = None
    collection: str | None = None

    @property
    def n_papers(self) -> int:
        return int(self.parse_report.get("n_papers", 0))

    def per_paper(self) -> list[dict[str, Any]]:
        """Per-paper rows, joined with the chunk counts the parse report cannot know."""
        by_paper: dict[str, int] = {}
        for chunk in self.chunks:
            by_paper[chunk.metadata.paper_id] = by_paper.get(chunk.metadata.paper_id, 0) + 1
        rows = []
        for row in self.parse_report.get("per_paper", []):
            rows.append({**row, "chunks": by_paper.get(row["paper_id"], 0)})
        return rows


def token_distribution(chunks: list[Chunk], tokenizer: str = "cl100k_base") -> dict[str, float]:
    """Summary of the chunk token distribution, the headline number of stage 02.

    Reports the quartiles rather than only the mean: a mean of 350 is equally consistent
    with a tight distribution and with a pile of runts next to a pile of cap-hitting
    chunks, and only the second is a problem.

    `n_tokens` isn't stored on the contract `Chunk` (Sude's shape carries `text` and
    `metadata` only), so it's recomputed here from the same tokenizer chunking used.
    """
    if not chunks:
        return {}
    enc = get_encoder(tokenizer)
    tokens = sorted(count_tokens(c.text, enc) for c in chunks)
    quartiles = statistics.quantiles(tokens, n=4) if len(tokens) > 1 else [tokens[0]] * 3
    return {
        "count": float(len(tokens)),
        "min": float(tokens[0]),
        "p25": float(round(quartiles[0], 1)),
        "median": float(round(statistics.median(tokens), 1)),
        "p75": float(round(quartiles[2], 1)),
        "max": float(tokens[-1]),
        "mean": float(round(statistics.fmean(tokens), 1)),
    }


def _should_run(stage: str, upto: str) -> bool:
    return _STAGE_ORDER[stage] <= _STAGE_ORDER[upto]


def run_ingestion(
    *,
    config: dict[str, Any] | None = None,
    stage: str = "all",
    limit: int | None = None,
    dry_run: bool = False,
    store: Any | None = None,
    keep_thin: bool = False,
    show_progress: bool = False,
) -> IngestionResult:
    """Run the pipeline up to `stage`.

    `dry_run` computes everything that is cheap and free of side effects -- parsing and
    chunking -- but writes no file, loads no model and touches no index. It is the check
    you run before a re-ingest to see how many chunks the new config would produce.

    `store` is injectable so tests can pass a fake satisfying the `VectorStore`
    interface instead of standing up Chroma.
    """
    if stage not in _STAGE_ORDER:
        raise ValueError(f"unknown stage {stage!r}, expected one of {STAGES}")
    cfg = config or load_config("ingestion")
    result = IngestionResult(stage=stage, dry_run=dry_run)

    pages, report = parse_corpus(config=cfg, limit=limit, keep_thin=keep_thin)
    result.pages = pages
    result.parse_report = report
    if not dry_run:
        interim = resolve_path(cfg["corpus"]["interim_dir"]) / "pages.jsonl"
        result.artefacts.append(write_pages(pages, interim))
    if not _should_run("chunk", stage):
        return result

    chunks = chunk_pages(pages, config=cfg)
    result.chunks = chunks
    result.token_stats = token_distribution(chunks, tokenizer=cfg["chunk"]["tokenizer"])
    if not dry_run:
        processed = resolve_path(cfg["corpus"]["processed_dir"]) / "chunks.jsonl"
        result.artefacts.append(write_chunks(chunks, processed))
        if chunks:
            # BM25 needs no embedding model, so it is built here rather than gated
            # behind the `embed` stage. Without this, `RetrievalService.bm25_index`
            # lazily loads whatever pickle happens to be on disk -- silently stale
            # after a re-chunk, or missing entirely on a fresh checkout, and the
            # sparse arm never actually participates in fusion either way.
            from research_assistant.retrieval.bm25_B import BM25Index

            result.artefacts.append(BM25Index.build(chunks).save())
    if not _should_run("embed", stage) or dry_run or not chunks:
        return result

    from research_assistant.ingestion.embed_B import embed_passages  # heavy; only if needed

    vectors = embed_passages([c.text for c in chunks], config=cfg, show_progress=show_progress)
    result.embedded = int(vectors.shape[0])
    result.embedding_dim = int(vectors.shape[1])
    if not _should_run("all", stage):
        return result

    if store is None:
        from research_assistant.retrieval.vector_store_B import VectorStore

        store = VectorStore()
    store.upsert(chunks, [v.tolist() for v in vectors])
    result.store_count = int(store.count())
    result.collection = getattr(store, "_collection_name", None) or type(store).__name__
    return result
