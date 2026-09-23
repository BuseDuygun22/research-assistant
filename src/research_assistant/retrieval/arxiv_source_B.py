"""arXiv search + download, shared by the offline fetch script and live discovery.

Extracted from `scripts/fetch_arxiv_fraud_B.py` rather than duplicated: that
script and `live_discovery_S.py` both need "search arXiv, get back candidate
papers, download the PDFs", and a `src/` module cannot import from `scripts/`
(a pip install of this package does not ship `scripts/`, so that import would
work in the repo and break in an installed deployment - `retrieval/` importing
`retrieval/`, on the other hand, always works).

httpx, not urllib: arXiv answers urllib's requests with HTTP 406 from some
networks (observed 2026-09-22) while accepting the identical request from httpx.
"""

from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

ARXIV_API = "https://export.arxiv.org/api/query"
HEADERS = {"User-Agent": "research-assistant-B/1.0 (arxiv source)"}
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# Scoped categories, not a bare keyword match — see fetch_arxiv_fraud_B.py's
# module docstring for why the domain boundary is written down rather than left
# to whatever a search term happens to match.
DEFAULT_CATEGORIES = ["cs.LG", "cs.CR", "cs.AI", "stat.ML", "q-fin.RM", "q-fin.ST"]


@dataclass(frozen=True)
class ArxivCandidate:
    arxiv_id: str
    title: str
    summary: str
    year: int
    category: str
    pdf_url: str

    @property
    def paper_id(self) -> str:
        """Same derivation as `fetch_arxiv_fraud_B._paper_id` - stable across
        both call sites so the same paper always gets the same id whichever
        path found it, and a live-discovered paper that matches one already in
        the static corpus is recognisably the same paper, not a duplicate."""
        return hashlib.sha1(self.arxiv_id.encode()).hexdigest()[:10]


def _fetch_page(query: str, start: int, max_results: int, *, timeout: float = 30) -> ET.Element:
    url = f"{ARXIV_API}?search_query={query}&start={start}&max_results={max_results}"
    resp = httpx.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return ET.fromstring(resp.content)


def _encode(term: str) -> str:
    """Space-to-`+`, nothing else. Deliberately does NOT auto-wrap a multi-word
    term in exact-phrase quotes: arXiv's query grammar treats a quoted phrase
    as "this exact wording appears verbatim", which a full reformulated
    question (live discovery's caller) or a natural-language search almost
    never does, and would silently return zero results every time. Bare,
    space-joined terms in an `all:` field are ANDed together, which is what a
    topical, question-derived query actually wants. A caller that *does* want
    an exact phrase (the static fetch script's "fraud detection") pre-encodes
    its own `%22...%22` and passes that through unchanged - this function
    never adds quoting on its own."""
    return term.strip().replace(" ", "+")


def build_query(terms: str, categories: Sequence[str]) -> str:
    """`terms` free text AND-ed with the category scope, mirroring
    `fetch_arxiv_fraud_B.CATEGORIES` restriction."""
    cat_clause = "+OR+".join(f"cat:{c}" for c in categories)
    return f"%28all:{_encode(terms)}%29+AND+%28{cat_clause}%29"


def search(
    terms: str,
    *,
    max_results: int,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    exclude_arxiv_ids: Collection[str] = (),
    page_size: int = 25,
    hard_stop: int = 200,
    etiquette_delay: float = 3.0,
) -> list[ArxivCandidate]:
    """Search arXiv, deduplicated and capped at `max_results`.

    `exclude_arxiv_ids` filters out papers already in a corpus (the static one,
    or ones already found earlier in the same discovery run) — the point of
    live discovery is *new* evidence, and re-finding a paper already indexed
    would spend the round's budget on nothing.
    """
    query = build_query(terms, categories)
    excluded = set(exclude_arxiv_ids)
    seen: set[str] = set()
    found: list[ArxivCandidate] = []
    start = 0
    while len(found) < max_results and start < hard_stop:
        root = _fetch_page(query, start, page_size)
        entries = root.findall("a:entry", NS)
        if not entries:
            break
        for e in entries:
            id_el = e.find("a:id", NS)
            if id_el is None or id_el.text is None:
                continue
            arxiv_id = re.sub(r"v\d+$", "", id_el.text.rsplit("/", 1)[-1])
            if arxiv_id in seen or arxiv_id in excluded:
                continue
            seen.add(arxiv_id)

            title_el, summary_el, pub_el = (
                e.find("a:title", NS),
                e.find("a:summary", NS),
                e.find("a:published", NS),
            )
            if title_el is None or title_el.text is None:
                continue
            if summary_el is None or summary_el.text is None:
                continue
            if pub_el is None or pub_el.text is None:
                continue

            pdf_link = next(
                (
                    link.get("href")
                    for link in e.findall("a:link", NS)
                    if link.get("title") == "pdf"
                ),
                None,
            )
            if pdf_link is None:
                continue

            primary_cat = e.find("arxiv:primary_category", NS)
            found.append(
                ArxivCandidate(
                    arxiv_id=arxiv_id,
                    title=" ".join(title_el.text.split()),
                    summary=" ".join(summary_el.text.split()),
                    year=int(pub_el.text[:4]),
                    category=primary_cat.get("term", "") if primary_cat is not None else "",
                    pdf_url=pdf_link if pdf_link.endswith(".pdf") else pdf_link + ".pdf",
                )
            )
            if len(found) >= max_results:
                break
        start += page_size
        if len(found) < max_results and start < hard_stop:
            time.sleep(etiquette_delay)  # arXiv API etiquette: <= 1 request / 3s
    return found


def download_pdf(candidate: ArxivCandidate, dest: Path, *, timeout: float = 60) -> None:
    resp = httpx.get(candidate.pdf_url, headers=HEADERS, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
