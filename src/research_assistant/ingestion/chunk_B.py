"""Stage 02 — page records to `Chunk` objects.

This stage sets the retrieval ceiling. A fact split across two chunks can never be
retrieved as one hit, no matter how good the embedder or the reranker is. The design
choices, the rejected strategies and the sweep live in `notebooks/02_chunking_B.ipynb`.

What is decided here, all of it driven by `configs/ingestion_B.yaml` under `chunk:`:

- **Section-aware grouping.** Consecutive records sharing a section become one unit
  before splitting, so a section that spans a page break is not cut at the break.
- **A token budget, not a character budget.** `target_tokens` with `overlap_tokens`
  carried forward, a `min_tokens` floor enforced by merging, and `max_tokens` as a hard
  cap honoured at a sentence boundary.
- **A `TITLE > SECTION` context prefix.** A chunk reading "we improve on this by four
  points" is unretrievable without knowing what "this" is. The cheap version of
  contextual retrieval.

Three places where the notebook prototype did not hold up, and what changed:

1. The prototype's overlap carried whole *paragraphs* backwards. On real paper prose
   every paragraph is larger than `overlap_tokens`, so the carry loop broke on its
   first iteration and produced zero overlap. Overlap now falls back to trailing
   sentences when no whole paragraph fits the budget.
2. The prototype could emit chunks above `max_tokens`, because a single paragraph
   longer than the cap was never split. Oversize paragraphs are now split at sentence
   boundaries, and a token-level split is the last resort for a runaway sentence.
3. The prototype measured the budget on the body text but stored body *plus* prefix,
   so the stored `n_tokens` could exceed the cap it was checked against. The prefix
   cost is now reserved out of the budget up front.

Output conforms to `contracts/retrieval_J.Chunk`, which is the boundary Sude's MCP
tools read. A missing field breaks her side, not ours.

Every piece of text carries its `(start, end)` offset into the section's original text
from the moment it is split out, and that span survives packing, overlap-carrying and
oversize-splitting unchanged. This is not cosmetic: `Chunk.derive_id` hashes
`char_start`/`char_end` directly, so a chunk_id that drifted from the real offset would
still be internally consistent but would no longer point at a checkable span in the
source document — which is the property `char_start`/`char_end` exist for in the first
place (citation, not just identity).
"""

from __future__ import annotations

import difflib
import functools
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tiktoken

from research_assistant.config_J import load_config, resolve_path
from research_assistant.contracts.retrieval_J import Chunk, ChunkMetadata
from research_assistant.ingestion.parse_B import PageRecord, load_manifest

# Paragraph break: a blank line, or a newline followed by a capital. The second form is
# needed because PyMuPDF joins block text with single newlines and never emits the blank
# line a paragraph break would have in plain text.
PARA_SPLIT = re.compile(r"\n\s*\n|\n(?=[A-Z])")

# Sentence break: terminal punctuation followed by whitespace. Deliberately naive --
# the failure mode (splitting after "et al.") costs a slightly odd boundary, not a lost
# fact, and a real sentence segmenter is a dependency this stage does not need.
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

CONTEXT_SEPARATOR = " > "

# One piece of text plus its (start, end) offset into the *section's* original text
# (unit.text), i.e. before any prefix is prepended. Threaded through every splitting
# and packing step so the final chunk's span is never re-derived from the (already
# whitespace-normalised) packed string, only ever narrowed from real offsets.
Span = tuple[str, int, int]

_TITLE_NOISE = re.compile(r"[^a-z0-9]+")


def _normalise_title(title: str) -> str:
    return _TITLE_NOISE.sub(" ", title.lower()).strip()


def canonical_paper_ids(
    manifest_rows: Iterable[dict[str, Any]], *, similarity: float = 0.92
) -> dict[str, str]:
    """Map each file's own `paper_id` to a canonical one, collapsing a preprint and
    its published version onto the same identity.

    Design review suggestion #8: Sude's side only deduplicates identical *text* at
    merge time (`agents/nodes/researcher_S.py`), which misses the normal case of a
    preprint and its revision differing by a sentence or two while being the same
    paper. Linking them here, by title, is the fix that catches that case instead
    of only the exact-duplicate one.

    Deterministic: rows are visited in `paper_id` order, and a title within
    `similarity` of an already-seen one is assigned that paper's id rather than a
    new one, so the same corpus re-ingested twice canonicalises the same way.
    `difflib.SequenceMatcher` is used rather than an embedding call — this runs at
    ingest, over every pair of titles, and a paper's title changing enough to drop
    below the threshold across a revision would be a real title change, not the
    kind of duplicate this guards against.
    """
    rows = sorted(manifest_rows, key=lambda r: str(r.get("paper_id", "")))
    canon: dict[str, str] = {}
    seen: list[tuple[str, str]] = []  # (normalised_title, canonical_paper_id)
    for row in rows:
        paper_id = str(row.get("paper_id", ""))
        title = _normalise_title(str(row.get("title", "")))
        match = next(
            (
                cid
                for norm, cid in seen
                if title and difflib.SequenceMatcher(None, title, norm).ratio() >= similarity
            ),
            None,
        )
        if match is not None:
            canon[paper_id] = match
        else:
            canon[paper_id] = paper_id
            seen.append((title, paper_id))
    return canon


@dataclass
class SectionUnit:
    """Consecutive page records sharing one section, re-joined before splitting."""

    paper_id: str
    section: str
    text: str
    page_start: int
    page_end: int
    title: str = ""
    pages: list[int] = field(default_factory=list)
    # (offset into `text`, page number) for every page folded into this unit, in order.
    # `ChunkMetadata.page` is a single int, not a range, so a chunk carved out of a
    # unit that spans several pages needs to look up *its own* page from its char
    # offset rather than inherit the unit's first page — otherwise every chunk from a
    # multi-page section is attributed to the section's opening page regardless of
    # where in it the chunk's text actually falls.
    page_offsets: list[tuple[int, int]] = field(default_factory=list)

    def page_at(self, offset: int) -> int:
        page = self.page_start
        for start, pg in self.page_offsets:
            if start > offset:
                break
            page = pg
        return page


@functools.lru_cache(maxsize=4)
def get_encoder(name: str) -> tiktoken.Encoding:
    """Cache the tiktoken encoding; constructing one is slow and it is thread-safe to share."""
    return tiktoken.get_encoding(name)


def count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def split_paragraphs(text: str) -> list[str]:
    """Split into paragraphs, never returning an empty list."""
    parts = [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _spanned(pieces: list[str], source: str, base: int = 0) -> list[Span]:
    """Locate each piece in `source`, in order, and return it with its offset.

    Pieces are produced by stripping whitespace off a regex split of `source`, so each
    one is guaranteed to occur in `source` at or after where the previous piece ended --
    that monotonic guarantee is what makes a plain sequential `find` exact rather than
    approximate. `base` shifts the result into an enclosing text's coordinate space, so
    sentence spans found inside one paragraph can be reported relative to the section.
    """
    spans: list[Span] = []
    cursor = 0
    for piece in pieces:
        idx = source.find(piece, cursor)
        if idx == -1:  # pragma: no cover - defensive; strip() only removes, never alters
            idx = cursor
        spans.append((piece, base + idx, base + idx + len(piece)))
        cursor = idx + len(piece)
    return spans


def _hard_split(span: Span, max_tokens: int, enc: tiktoken.Encoding) -> list[Span]:
    """Last-resort token-level split for a single sentence longer than the cap.

    This should essentially never fire on prose. It exists so that the `max_tokens`
    guarantee is unconditional rather than "unless the document is strange", because a
    chunk over the cap silently truncates inside the embedding model.

    Token decode does not round-trip to an exact character offset (a token can straddle
    a character that whitespace-normalises on decode), so the offsets here are
    proportional estimates over the sentence's real span rather than exact. Deterministic
    and monotonic either way, which is what `derive_id` needs; only this unreachable-in-
    practice path pays the approximation.
    """
    text, start, end = span
    ids = enc.encode(text)
    total = len(ids)
    pieces = [enc.decode(ids[i : i + max_tokens]) for i in range(0, total, max_tokens)]
    span_len = end - start
    out: list[Span] = []
    consumed = 0
    for i, piece in enumerate(pieces):
        frac_start = consumed / total if total else 0
        consumed += min(max_tokens, total - i * max_tokens)
        frac_end = consumed / total if total else 1
        piece_start = start + round(frac_start * span_len)
        piece_end = start + round(frac_end * span_len)
        out.append((piece, piece_start, piece_end))
    return out


def _split_oversize(span: Span, target: int, max_tokens: int, enc: tiktoken.Encoding) -> list[Span]:
    """Break a paragraph that exceeds the cap into sentence groups that do not."""
    text, start, _end = span
    sentences = _spanned(split_sentences(text), text, base=start)
    pieces: list[Span] = []
    buf: list[Span] = []
    buf_tok = 0
    for sentence in sentences:
        stok = count_tokens(sentence[0], enc)
        if stok > max_tokens:
            if buf:
                pieces.append((" ".join(s[0] for s in buf), buf[0][1], buf[-1][2]))
                buf, buf_tok = [], 0
            pieces.extend(_hard_split(sentence, max_tokens, enc))
            continue
        if buf and buf_tok + stok > target:
            pieces.append((" ".join(s[0] for s in buf), buf[0][1], buf[-1][2]))
            buf, buf_tok = [], 0
        buf.append(sentence)
        buf_tok += stok
    if buf:
        pieces.append((" ".join(s[0] for s in buf), buf[0][1], buf[-1][2]))
    return pieces


def _tail_overlap(buf: list[Span], overlap: int, enc: tiktoken.Encoding) -> list[Span]:
    """Pick the trailing text to carry into the next chunk, up to `overlap` tokens.

    Whole pieces are preferred because they are self-contained. When even the last piece
    alone is over budget -- the normal case for paper paragraphs against a 60-token
    overlap -- fall back to its trailing sentences. The notebook version stopped at the
    first branch and therefore produced no overlap at all on real prose.
    """
    if overlap <= 0 or not buf:
        return []
    tail: list[Span] = []
    tail_tok = 0
    for piece in reversed(buf):
        ptok = count_tokens(piece[0], enc)
        if tail_tok + ptok > overlap:
            break
        tail.insert(0, piece)
        tail_tok += ptok
    if tail:
        return tail

    text, start, _end = buf[-1]
    sentences = _spanned(split_sentences(text), text, base=start)
    for i in range(len(sentences) - 1, -1, -1):
        candidate = sentences[i:]
        if count_tokens(" ".join(s[0] for s in candidate), enc) > overlap:
            return sentences[i + 1 :] if i + 1 < len(sentences) else []
    return sentences


def pack_pieces(
    pieces: list[Span],
    *,
    target: int,
    overlap: int,
    min_tokens: int,
    max_tokens: int,
    enc: tiktoken.Encoding,
) -> list[Span]:
    """Greedily pack paragraphs into chunks, carrying overlap forward, then merge runts.

    The flush happens *before* appending the piece that would blow the target, so a chunk
    only ever exceeds `target` by the overlap it inherited plus one piece -- and the
    `max_tokens` guard drops the inherited overlap rather than breach the cap.
    """
    expanded: list[Span] = []
    for piece in pieces:
        if count_tokens(piece[0], enc) > max_tokens:
            expanded.extend(_split_oversize(piece, target, max_tokens, enc))
        else:
            expanded.append(piece)

    out: list[Span] = []
    buf: list[Span] = []
    buf_tok = 0
    for piece in expanded:
        ptok = count_tokens(piece[0], enc)
        if buf and buf_tok + ptok > target:
            out.append((" ".join(s[0] for s in buf), buf[0][1], buf[-1][2]))
            buf = _tail_overlap(buf, overlap, enc)
            buf_tok = count_tokens(" ".join(s[0] for s in buf), enc) if buf else 0
            if buf_tok + ptok > max_tokens:
                buf, buf_tok = [], 0
        buf.append(piece)
        buf_tok += ptok
    if buf:
        out.append((" ".join(s[0] for s in buf), buf[0][1], buf[-1][2]))

    merged: list[Span] = []
    for chunk in out:
        if merged and count_tokens(chunk[0], enc) < min_tokens:
            candidate_text = merged[-1][0] + " " + chunk[0]
            if count_tokens(candidate_text, enc) <= max_tokens:
                merged[-1] = (candidate_text, merged[-1][1], chunk[2])
                continue
        merged.append(chunk)
    return merged


def group_sections(pages: list[PageRecord]) -> list[SectionUnit]:
    """Group consecutive records sharing `(paper_id, section)` into one unit.

    Consecutive, not global: a section name that recurs later in the same paper (a
    heading detector misfire, or a genuinely repeated label) stays a separate unit, so
    text from page 2 is never glued to text from page 9 under one page range.
    """
    units: list[SectionUnit] = []
    current: SectionUnit | None = None
    for rec in pages:
        if current is not None and (current.paper_id, current.section) == (
            rec.paper_id,
            rec.section,
        ):
            current.page_offsets.append((len(current.text) + 1, rec.page))
            current.text += "\n" + rec.text
            current.page_end = max(current.page_end, rec.page)
            if rec.page not in current.pages:
                current.pages.append(rec.page)
            continue
        if current is not None:
            units.append(current)
        current = SectionUnit(
            paper_id=rec.paper_id,
            section=rec.section,
            text=rec.text,
            page_start=rec.page,
            page_end=rec.page,
            title=rec.title,
            pages=[rec.page],
            page_offsets=[(0, rec.page)],
        )
    if current is not None:
        units.append(current)
    return units


def _join_with_offsets(units: list[SectionUnit]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate units with `\\n`, carrying each one's page_offsets into the joined
    text's coordinate space. Mirrors the plain `"\\n".join(...)` used for `.text` so
    the two stay in lockstep -- an offset table that drifted from the text it
    describes would misattribute every chunk's page silently."""
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for u in units:
        entries = u.page_offsets or [(0, u.page_start)]
        offsets.extend((cursor + off, pg) for off, pg in entries)
        parts.append(u.text)
        cursor += len(u.text) + 1  # +1 for the "\n" joiner
    return "\n".join(parts), offsets


def absorb_small_units(
    units: list[SectionUnit], min_tokens: int, enc: tiktoken.Encoding
) -> list[SectionUnit]:
    """Fold a unit too small to be a chunk into its neighbour within the same paper.

    Front matter is the case that forces this: a title block plus a venue line is around
    twenty tokens, which no within-unit merge can rescue because it is the only piece in
    its unit. Attaching it to the *following* section is the semantically right move --
    a title belongs with the abstract that follows, not with the previous paper.
    """
    out: list[SectionUnit] = []
    pending: list[SectionUnit] = []
    for unit in units:
        if count_tokens(unit.text, enc) < min_tokens:
            pending.append(unit)
            continue
        if pending and pending[-1].paper_id == unit.paper_id:
            carried = [p for p in pending if p.paper_id == unit.paper_id]
            carried_text, carried_offsets = _join_with_offsets(carried)
            prefix_len = len(carried_text) + 1
            unit.page_offsets = carried_offsets + [
                (off + prefix_len, pg) for off, pg in (unit.page_offsets or [(0, unit.page_start)])
            ]
            unit.text = carried_text + "\n" + unit.text
            unit.page_start = min(unit.page_start, *(p.page_start for p in carried))
            unit.title = unit.title or next((p.title for p in carried if p.title), "")
        elif pending and out and out[-1].paper_id == pending[-1].paper_id:
            prefix_len = len(out[-1].text) + 1
            appended_text, appended_offsets = _join_with_offsets(pending)
            out[-1].page_offsets = (out[-1].page_offsets or [(0, out[-1].page_start)]) + [
                (off + prefix_len, pg) for off, pg in appended_offsets
            ]
            out[-1].text += "\n" + appended_text
            out[-1].page_end = max(out[-1].page_end, *(p.page_end for p in pending))
        pending = []
        out.append(unit)
    if pending and out and out[-1].paper_id == pending[-1].paper_id:
        prefix_len = len(out[-1].text) + 1
        appended_text, appended_offsets = _join_with_offsets(pending)
        out[-1].page_offsets = (out[-1].page_offsets or [(0, out[-1].page_start)]) + [
            (off + prefix_len, pg) for off, pg in appended_offsets
        ]
        out[-1].text += "\n" + appended_text
        out[-1].page_end = max(out[-1].page_end, *(p.page_end for p in pending))
    elif pending:
        out.extend(pending)
    return out


def build_chunks(
    units: list[SectionUnit],
    chunk_cfg: dict[str, Any],
    titles: dict[str, str] | None = None,
    metadata: dict[str, dict[str, Any]] | None = None,
    canonical_ids: dict[str, str] | None = None,
) -> list[Chunk]:
    """Turn section units into contract-valid `Chunk` objects.

    The context prefix is charged against the token budget before packing, so the stored
    `n_tokens` -- which includes the prefix, because the prefix is what gets embedded --
    still respects `max_tokens`. The prefix is never part of `char_start`/`char_end`: it
    is manufactured here, not present in the source document, so a citation built from
    those offsets must stay resolvable against the paper's own text.

    `canonical_ids` (see `canonical_paper_ids`) maps a file's own paper_id to the id a
    preprint and its published version should share. It only relabels
    `ChunkMetadata.paper_id`; `source_uri` still points at the actual file a chunk's
    text came from, so a citation remains resolvable even when two files share an
    identity.
    """
    enc = get_encoder(chunk_cfg["tokenizer"])
    titles = titles or {}
    metadata = metadata or {}
    canonical_ids = canonical_ids or {}
    target = int(chunk_cfg["target_tokens"])
    overlap = int(chunk_cfg["overlap_tokens"])
    min_tokens = int(chunk_cfg["min_tokens"])
    max_tokens = int(chunk_cfg["max_tokens"])
    prepend = bool(chunk_cfg.get("prepend_context", True))
    respect_paragraphs = bool(chunk_cfg.get("respect_paragraphs", True))

    chunks: list[Chunk] = []
    for unit in units:
        title = titles.get(unit.paper_id, unit.title) or unit.title
        paper_id = canonical_ids.get(unit.paper_id, unit.paper_id)
        prefix = f"{title}{CONTEXT_SEPARATOR}{unit.section}\n" if prepend else ""
        prefix_tokens = count_tokens(prefix, enc) if prefix else 0
        budget_target = max(target - prefix_tokens, min_tokens)
        budget_max = max(max_tokens - prefix_tokens, budget_target)

        raw_pieces = split_paragraphs(unit.text) if respect_paragraphs else [unit.text]
        pieces = _spanned(raw_pieces, unit.text)
        bodies = pack_pieces(
            pieces,
            target=budget_target,
            overlap=overlap,
            min_tokens=min_tokens,
            max_tokens=budget_max,
            enc=enc,
        )
        meta = metadata.get(unit.paper_id, {})
        source_uri = meta.get("source_uri") or (
            f"data/raw/{meta['filename']}" if meta.get("filename") else None
        )
        for body, char_start, char_end in bodies:
            text = prefix + body
            chunk_meta = ChunkMetadata(
                paper_id=paper_id,
                title=title,
                authors=list(meta.get("authors", [])),
                year=meta.get("year"),
                venue=meta.get("venue"),
                section=unit.section,
                page=unit.page_at(char_start),
                char_start=char_start,
                char_end=char_end,
                source_uri=source_uri,
            )
            chunks.append(
                Chunk(
                    chunk_id=Chunk.derive_id(paper_id, char_start, char_end, text),
                    text=text,
                    metadata=chunk_meta,
                )
            )
    return chunks


def chunk_pages(
    pages: list[PageRecord],
    *,
    config: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Top-level stage-02 entry point: page records in, contract-valid chunks out."""
    cfg = config or load_config("ingestion")
    chunk_cfg = cfg["chunk"]
    enc = get_encoder(chunk_cfg["tokenizer"])

    manifest = load_manifest(resolve_path(cfg["corpus"]["manifest"]))
    titles = {row["paper_id"]: row.get("title", "") for row in manifest.values()}
    metadata = {row["paper_id"]: row for row in manifest.values()}
    canonical_ids = canonical_paper_ids(manifest.values())

    units = group_sections(pages)
    units = absorb_small_units(units, int(chunk_cfg["min_tokens"]), enc)
    return build_chunks(units, chunk_cfg, titles, metadata, canonical_ids)


def write_chunks(chunks: list[Chunk], out_path: str | Path) -> Path:
    """Persist chunks as JSONL, the stage-02 handoff artefact."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return path
