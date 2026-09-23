"""Opt-in live paper discovery (Sude): expand the corpus when it cannot answer.

Off by default (`RA_ALLOW_LIVE_DISCOVERY=false`). The default closed corpus is
what makes every citation in this system trustworthy by construction - a paper
that went through `search`, `parse_corpus`, `chunk_pages` and `embed_passages`
here has been through the same pipeline as the other 30, with real page numbers
and real chunk boundaries, unlike a paper cited straight off a live web search.
Opting in trades some of that guarantee for coverage a fixed 30-paper corpus
cannot have. Every live-discovered chunk's `venue` is suffixed
`[live discovery]` so a reader can tell which citations went through the full
review the static corpus did and which did not.

**Isolation, deliberately incomplete.** Discovered papers are parsed, chunked
and embedded into a session-scoped Chroma collection and an in-memory BM25
index (`data/live_discovery/<run_id>/`), never into the audited `papers_v2`
collection or `data/bm25_index_B.pkl` - a live-discovery run can never corrupt
the fixed corpus, and `corpus_version` keeps meaning exactly what it always
meant. What is *not* isolated: a successful discovery calls `set_backend()`
with a backend that unions the fixed and live indexes, which is a process-wide
change for the rest of the process's life. That is fine for `ask_S` (one
question per process) and a single-developer local server session; it is not
safe for a shared multi-tenant deployment, where one user's discovery would
leak into another's queries. Documented here rather than silently assumed.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from research_assistant.config_J import REPO_ROOT, load_config
from research_assistant.contracts.retrieval_J import (
    Chunk,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)
from research_assistant.mcp_server.backend_S import RetrievalBackend
from research_assistant.retrieval.arxiv_source_B import (
    DEFAULT_CATEGORIES,
    ArxivCandidate,
    download_pdf,
)
from research_assistant.retrieval.arxiv_source_B import search as search_arxiv

logger = logging.getLogger(__name__)

LIVE_DISCOVERY_ROOT = REPO_ROOT / "data" / "live_discovery"
_ARXIV_VENUE = re.compile(r"arXiv:(\S+)")


@dataclass(frozen=True)
class DiscoveryOutcome:
    """What one discovery attempt produced. Always returned, never raised - a
    failed discovery (no network, no candidates, a parse error on every PDF)
    degrades to `backend=None`, which the caller treats exactly like "found
    nothing" rather than crashing the run over an optional feature."""

    papers: list[ArxivCandidate]
    backend: RetrievalBackend | None
    error: str | None = None

    @property
    def found_anything(self) -> bool:
        return self.backend is not None


class UnionRetrieval:
    """Two backends queried as one. Not a third retrieval implementation - a
    thin merge of results from an already-bound `primary` (the fixed corpus)
    and a `secondary` (a live-discovery index), so the rest of the pipeline
    (triage, writer, citations) never has to know a run used two indexes."""

    def __init__(self, primary: RetrievalBackend, secondary: RetrievalBackend) -> None:
        self._primary = primary
        self._secondary = secondary

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        a = self._primary.retrieve(request)
        b = self._secondary.retrieve(request)
        by_id = {r.chunk.chunk_id: r for r in (*a.results, *b.results)}
        ordered = sorted(by_id.values(), key=lambda r: r.score, reverse=True)[: request.top_k]
        # Each arm ranked its own results 1..n independently; re-number after
        # merging, or two chunks both carrying "rank=1" would misreport the
        # combined order every consumer (citations, eval) trusts `rank` for.
        merged: list[RetrievedChunk] = [
            r.model_copy(update={"rank": i}) for i, r in enumerate(ordered, start=1)
        ]
        return RetrievalResponse(
            query=request.query,
            results=merged,
            corpus_version=f"{a.corpus_version}+live",
            embedding_model=a.embedding_model,
            reranker_version=a.reranker_version,
        )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._primary.get_chunk(chunk_id) or self._secondary.get_chunk(chunk_id)


def _known_arxiv_ids(manifest_path: Path) -> set[str]:
    """arXiv ids already in the static corpus, parsed out of `venue`
    ("arXiv:1234.5678 (cs.LG)") - live discovery re-finding a paper already
    indexed would spend the round's budget on nothing new."""
    if not manifest_path.exists():
        return set()
    ids: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = _ARXIV_VENUE.search(json.loads(line).get("venue") or "")
        if m:
            ids.add(m.group(1))
    return ids


def _live_ingestion_config(run_dir: Path) -> dict[str, Any]:
    """The real ingestion config, with only the corpus paths redirected under
    `run_dir` - same chunking/embedding parameters as the audited pipeline, so
    a live-discovered chunk is comparable to a static one, not a second recipe."""
    cfg = copy.deepcopy(load_config("ingestion"))
    cfg["corpus"] = {
        **cfg["corpus"],
        "raw_dir": str(run_dir / "raw"),
        "interim_dir": str(run_dir / "interim"),
        "processed_dir": str(run_dir / "processed"),
        "manifest": str(run_dir / "manifest.jsonl"),
    }
    return cfg


def _index_candidates(
    papers: list[ArxivCandidate], run_dir: Path, run_id: str
) -> RetrievalBackend:
    """Download, parse, chunk, embed and index `papers` into a session-scoped
    Chroma collection + in-memory BM25 index. Composed from the same primitives
    `ingest_B.py` uses, deliberately *not* via `run_ingestion()`: that helper
    always saves the BM25 sidecar to the shared `data/bm25_index_B.pkl`, with
    no path override, and calling it here would silently corrupt the audited
    corpus's index on every live-discovery run."""
    from research_assistant.ingestion.chunk_B import chunk_pages
    from research_assistant.ingestion.embed_B import embed_passages
    from research_assistant.ingestion.parse_B import parse_corpus
    from research_assistant.retrieval.bm25_B import BM25Index
    from research_assistant.retrieval.service_B import RetrievalService
    from research_assistant.retrieval.vector_store_B import VectorStore

    raw_dir = run_dir / "raw"
    manifest_rows = []
    for p in papers:
        try:
            download_pdf(p, raw_dir / f"{p.paper_id}.pdf")
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("live discovery: failed to download %s (%s)", p.arxiv_id, exc)
            continue
        manifest_rows.append(
            {
                "paper_id": p.paper_id,
                "filename": f"{p.paper_id}.pdf",
                "title": p.title,
                "year": p.year,
                # Marked so `format_citation` and any reader can tell this chunk
                # did not go through the same review as the static corpus.
                "venue": f"arXiv:{p.arxiv_id} ({p.category}) [live discovery]",
                "why_included": f"live-discovered; abstract: {p.summary[:180]}...",
            }
        )

    cfg = _live_ingestion_config(run_dir)
    (run_dir / "manifest.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest_rows) + "\n",
        encoding="utf-8",
    )

    pages, _report = parse_corpus(config=cfg)
    chunks = chunk_pages(pages, config=cfg)
    if not chunks:
        raise ValueError("no chunks produced from the downloaded PDFs")

    vectors = embed_passages([c.text for c in chunks], config=cfg)
    store = VectorStore(persist_dir=run_dir / "chroma", collection=f"live-{run_id}")
    store.upsert(chunks, [v.tolist() for v in vectors])
    bm25 = BM25Index.build(chunks)

    return RetrievalService(vector_store=store, bm25_index=bm25, config=load_config("retrieval"))


def discover(
    query: str,
    *,
    run_id: str,
    max_papers: int = 8,
    categories: list[str] | None = None,
) -> DiscoveryOutcome:
    """Search arXiv for `query`, index what it finds, return a backend for it.

    Never raises: a network failure, an empty result set, or a parse failure on
    every candidate all come back as `DiscoveryOutcome(backend=None, error=...)`
    so the caller's fallback (fall through to the normal insufficient-evidence
    routing) always runs.
    """
    manifest_path = REPO_ROOT / "data" / "corpus_manifest_B.jsonl"
    exclude = _known_arxiv_ids(manifest_path)
    try:
        papers = search_arxiv(
            query,
            max_results=max_papers,
            categories=categories or DEFAULT_CATEGORIES,
            exclude_arxiv_ids=exclude,
        )
    except httpx.HTTPError as exc:
        logger.warning("live discovery: arXiv search failed (%s)", exc)
        return DiscoveryOutcome(papers=[], backend=None, error=str(exc))

    if not papers:
        return DiscoveryOutcome(papers=[], backend=None, error="no new candidates found")

    run_dir = LIVE_DISCOVERY_ROOT / run_id
    try:
        backend = _index_candidates(papers, run_dir, run_id)
    except Exception as exc:  # noqa: BLE001 - an optional feature must not crash the run
        logger.exception("live discovery: indexing failed")
        return DiscoveryOutcome(papers=papers, backend=None, error=str(exc))

    logger.info(
        "live discovery: indexed %d paper(s) for run %s: %s",
        len(papers),
        run_id,
        [p.arxiv_id for p in papers],
    )
    return DiscoveryOutcome(papers=papers, backend=backend)
