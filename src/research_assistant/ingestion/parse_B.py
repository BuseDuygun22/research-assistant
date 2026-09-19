"""Stage 01 — PDF to page records.

Decides what information can exist downstream. A page number lost here is a citation
that stage 06 cannot verify; a heading missed here silently downgrades section-aware
chunking to fixed windows in stage 02. Reasoning and the rejected alternatives are in
`notebooks/01_parsing_B.ipynb`.

Two choices from that notebook are load-bearing and are preserved here:

- **The unit of output is a page, not a document and not a block.** Keeping a
  page-level intermediate means every chunking experiment in stage 02 re-runs without
  re-parsing, and it is what makes page-range citations possible at all.
- **Thin pages are flagged, never silently dropped.** A page that yields 40 characters
  is a corpus problem you need to see. `PageRecord.thin` carries that signal forward;
  filtering is the caller's decision, made explicit via `keep_thin`.

Every threshold comes from `configs/ingestion_B.yaml` under `parse:`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pymupdf  # the `fitz` alias is deprecated

from research_assistant.config_J import load_config, resolve_path

# "1 Introduction", "3.2 Ablations", "4. Results" -- a numeric prefix followed by a
# capitalised word. Requiring the capital is what stops it firing on "2019 saw ...".
HEADING_NUM = re.compile(r"^\s*(\d+(\.\d+)*)[\.\)]?\s+[A-Z]")

# A real word: 3+ Latin letters in a row. Display/inline math set in a large or bold
# font ("v ∈ V p ( u )", "0 0 0 0") otherwise clears the size/bold test on every
# page of a math-heavy paper and gets misread as a heading. Discovered on the
# fraud-detection arXiv corpus, where several GNN papers produced 100+ "sections" per
# paper before this gate: nearly all of them were fragments of an equation.
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
MIN_ALPHA_RATIO = 0.5

FRONT_MATTER = "FRONT_MATTER"

#: Fallback body font size for a document whose spans carry no usable size info.
DEFAULT_BODY_SIZE = 10.0

#: A heading is short. These bound the heuristic so a full justified paragraph that
#: happens to start with a number can never be mistaken for a section title.
MAX_HEADING_CHARS = 90
MAX_HEADING_WORDS = 12

#: A heading must clear the body font by this much to qualify on size alone. Smaller
#: than a typical heading step (2-4pt) but larger than the sub-point jitter PyMuPDF
#: reports within a single body paragraph.
HEADING_SIZE_MARGIN = 0.8


@dataclass
class PageRecord:
    """One section segment of one page, with the section it belongs to.

    The notebook prototype emitted exactly one record per page, labelled with the last
    heading seen on that page. That loses every heading but the last on a dense page,
    and it makes the `drop_sections` filter page-granular: a "References" heading two
    thirds of the way down a page would discard the Results text above it. Segmenting
    at heading boundaries fixes both while keeping the page number intact, which is the
    property the rest of the pipeline actually depends on. `ordinal` preserves reading
    order for segments sharing a page.

    `dropped` and `thin` are advisory flags rather than filters so that the count of
    what was discarded stays visible in the pipeline summary instead of vanishing.
    """

    paper_id: str
    page: int
    section: str
    text: str
    n_chars: int
    dropped: bool
    thin: bool
    ordinal: int = 0
    source: str = ""
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def page_blocks(page: pymupdf.Page) -> list[tuple[str, float, bool]]:
    """Return `(text, max_font_size, is_bold)` per text block, in reading order.

    Font size is taken as the *maximum* span size in the block, not the mean: a heading
    block often carries a trailing small-caps or footnote-marker span that would drag a
    mean below the detection margin.
    """
    out: list[tuple[str, float, bool]] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # 0 == text; images and drawings carry no headings
            continue
        parts: list[str] = []
        size = 0.0
        bold = False
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                parts.append(span["text"])
                size = max(size, float(span["size"]))
                bold = bold or "bold" in span["font"].lower()
        joined = " ".join(parts).strip()
        if joined:
            out.append((joined, size, bold))
    return out


def looks_like_heading(
    text: str,
    size: float,
    bold: bool,
    body_size: float,
    *,
    max_chars: int = MAX_HEADING_CHARS,
    max_words: int = MAX_HEADING_WORDS,
    size_margin: float = HEADING_SIZE_MARGIN,
) -> bool:
    """Decide whether a block is a section heading, relative to the document's body font.

    The length gate is applied *before* the numbering test on purpose. A numbered
    enumeration inside a paragraph ("2. we then re-rank the candidates and ...") matches
    the numbering regex, so length is the only thing separating it from a real heading.

    Numbering alone is not sufficient, even past the length gate: a numbered
    bibliography entry ("41. Ashtiani M N, Raahemi B. Intelligent fraud ...") is short,
    numbered and capitalised, and would otherwise pass unconditionally. Discovered on
    the arXiv fraud-detection corpus, where an undetected References heading let every
    citation line become its own spurious "section" -- 100+ per paper. A real numbered
    heading is reliably distinguished from body-font citation text by weight or size, so
    the numbering branch now requires that too, same as the unnumbered branch below.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > max_chars or len(stripped.split()) > max_words:
        return False
    if HEADING_NUM.match(stripped):
        return size > body_size + size_margin or (bold and size >= body_size)
    letters = sum(1 for ch in stripped if ch.isalpha())
    if letters / len(stripped) < MIN_ALPHA_RATIO or not _WORD_RE.search(stripped):
        return False  # a heading is prose; an equation fragment is not
    return size > body_size + size_margin or (bold and size >= body_size)


def estimate_body_size(doc: pymupdf.Document) -> float:
    """Modal span size across the document.

    Modal rather than mean or median: a paper is overwhelmingly body text, so the mode
    is the body font almost by definition, while a mean is pulled up by the title page
    and a median shifts on documents that are mostly tables.
    """
    sizes: Counter[float] = Counter()
    for page in doc:
        for _text, size, _bold in page_blocks(page):
            sizes[round(size, 1)] += 1
    return max(sizes, key=lambda s: sizes[s]) if sizes else DEFAULT_BODY_SIZE


def parse_pdf(
    path: str | Path,
    paper_id: str,
    drop_sections: list[str] | None = None,
    *,
    min_chars_per_page: int = 200,
    title: str = "",
) -> tuple[list[PageRecord], float]:
    """Parse one PDF into page records. Returns the records and the detected body size.

    The current section persists across pages: a section heading seen on page 4 governs
    page 5 too, until the next heading. That is what lets stage 02 group consecutive
    same-section records back into one unit spanning the page break.

    Thinness is judged on the whole page, not the segment. A short segment is normal --
    a heading landing at the foot of a page produces one. A short *page* is the signal
    the config's `min_chars_per_page` is actually about: a scan or a full-page figure.
    """
    path = Path(path)
    drops = [d.lower() for d in (drop_sections or [])]
    doc = pymupdf.open(path)
    try:
        body_size = estimate_body_size(doc)
        records: list[PageRecord] = []
        current_section = FRONT_MATTER
        for page_no, page in enumerate(doc, start=1):
            segments: list[tuple[str, list[str]]] = [(current_section, [])]
            page_chars = 0
            for text, size, bold in page_blocks(page):
                if looks_like_heading(text, size, bold, body_size):
                    current_section = text.strip()
                    segments.append((current_section, []))
                segments[-1][1].append(text)
                page_chars += len(text)

            page_is_thin = page_chars < min_chars_per_page
            ordinal = 0
            for section, parts in segments:
                body = "\n".join(parts).strip()
                if not body:
                    continue
                low = section.lower()
                records.append(
                    PageRecord(
                        paper_id=paper_id,
                        page=page_no,
                        section=section,
                        text=body,
                        n_chars=len(body),
                        dropped=any(d in low for d in drops),
                        thin=page_is_thin,
                        ordinal=ordinal,
                        source=path.name,
                        title=title,
                    )
                )
                ordinal += 1
    finally:
        doc.close()
    return records, body_size


def load_manifest(manifest_path: str | Path) -> dict[str, dict[str, Any]]:
    """Read the corpus manifest, keyed by filename. Missing manifest is not an error.

    Absence is tolerated so that a directory of PDFs can be parsed before anyone has
    written the manifest; paper ids then fall back to the filename stem.
    """
    path = Path(manifest_path)
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["filename"]] = row
    return rows


def parse_corpus(
    raw_dir: str | Path | None = None,
    *,
    config: dict[str, Any] | None = None,
    limit: int | None = None,
    keep_thin: bool = False,
) -> tuple[list[PageRecord], dict[str, Any]]:
    """Parse every PDF in the raw directory. Returns kept page records plus a report.

    `keep_thin=False` filters thin pages out of the returned records but still counts
    them in the report, because the notebook's exit check is that you *look* at them.
    """
    cfg = config or load_config("ingestion")
    parse_cfg = cfg["parse"]
    corpus_cfg = cfg["corpus"]
    raw = Path(raw_dir) if raw_dir is not None else resolve_path(corpus_cfg["raw_dir"])
    manifest = load_manifest(resolve_path(corpus_cfg["manifest"]))
    min_chars = int(parse_cfg["min_chars_per_page"])
    drops = list(parse_cfg.get("drop_sections") or [])

    pdfs = sorted(raw.glob("*.pdf"))
    if limit is not None:
        pdfs = pdfs[:limit]

    kept: list[PageRecord] = []
    per_paper: list[dict[str, Any]] = []
    thin_pages: list[dict[str, Any]] = []
    total_pages = 0
    dropped_records = 0

    for pdf in pdfs:
        meta = manifest.get(pdf.name, {})
        paper_id = meta.get("paper_id") or pdf.stem[:10]
        records, body_size = parse_pdf(
            pdf,
            paper_id,
            drops,
            min_chars_per_page=min_chars,
            title=meta.get("title", ""),
        )
        paper_pages = {r.page for r in records}
        total_pages += len(paper_pages)
        seen_thin: set[int] = set()
        for rec in records:
            if rec.thin and not rec.dropped and rec.page not in seen_thin:
                seen_thin.add(rec.page)
                thin_pages.append({"file": pdf.name, "page": rec.page, "n_chars": rec.n_chars})
            if rec.dropped or (rec.thin and not keep_thin):
                dropped_records += 1
                continue
            kept.append(rec)
        kept_here = [r for r in records if not r.dropped and (keep_thin or not r.thin)]
        sections = {r.section for r in kept_here}
        per_paper.append(
            {
                "file": pdf.name,
                "paper_id": paper_id,
                "title": meta.get("title", ""),
                "pages": len(paper_pages),
                "pages_kept": len({r.page for r in kept_here}),
                "records": len(records),
                "records_kept": len(kept_here),
                "body_font": body_size,
                "sections": len(sections),
                "section_names": sorted(sections),
            }
        )

    report = {
        "n_papers": len(pdfs),
        "total_pages": total_pages,
        "pages_kept": len({(r.paper_id, r.page) for r in kept}),
        "records_kept": len(kept),
        "records_dropped": dropped_records,
        "thin_pages": thin_pages,
        "per_paper": per_paper,
    }
    return kept, report


def write_pages(records: list[PageRecord], out_path: str | Path) -> Path:
    """Persist page records as JSONL, the stage-01 handoff artefact."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return path
