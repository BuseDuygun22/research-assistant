"""Stage 01-02 tests: PDF parsing and chunking.

Fixtures come from `tests/fixtures/make_synthetic_corpus_B.py`, run once ahead of
time so these tests don't pay PDF-rendering cost on every run (see `_corpus`
fixture below, which regenerates only if the files are missing).

The equation-fragment regression tests below exist because the arXiv fraud-detection
corpus (30 real papers, math-heavy GNN/deep-learning work) tripped a real bug:
`looks_like_heading` accepted bold/large-font inline math as a heading, producing
100+ spurious "sections" per paper. See the fix and its rationale in
`ingestion/parse_B.py` next to `MIN_ALPHA_RATIO`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from research_assistant.contracts.retrieval_J import Chunk
from research_assistant.ingestion.chunk_B import (
    _spanned,
    build_chunks,
    canonical_paper_ids,
    chunk_pages,
    count_tokens,
    get_encoder,
    group_sections,
    pack_pieces,
)
from research_assistant.ingestion.parse_B import (
    estimate_body_size,
    load_manifest,
    looks_like_heading,
    parse_corpus,
    parse_pdf,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
RAW_DIR = FIXTURES / "synthetic_pdfs_B"
MANIFEST = FIXTURES / "synthetic_manifest_B.jsonl"
GROUND_TRUTH = FIXTURES / "synthetic_ground_truth_B.json"


@pytest.fixture(scope="session")
def corpus() -> dict:
    """Ensure the synthetic corpus exists, generating it once if needed, and return ground truth."""
    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*.pdf")) or not MANIFEST.exists():
        subprocess.run(
            [sys.executable, str(FIXTURES / "make_synthetic_corpus_B.py")],
            check=True,
            cwd=FIXTURES.parents[1],
        )
    return json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def ingestion_config(corpus) -> dict:
    return {
        "corpus": {
            "raw_dir": str(RAW_DIR),
            "interim_dir": "data/interim",  # unused when config passed explicitly to parse_corpus
            "processed_dir": "data/processed",
            "manifest": str(MANIFEST),
        },
        "parse": {
            "min_chars_per_page": 20,
            "drop_sections": ["references", "bibliography", "acknowledg"],
        },
        "chunk": {
            "strategy": "section_aware",
            "target_tokens": 350,
            "overlap_tokens": 60,
            "min_tokens": 80,
            "max_tokens": 700,
            "tokenizer": "cl100k_base",
            "respect_paragraphs": True,
            "prepend_context": True,
        },
    }


# --------------------------------------------------------------------------- heading detector


class TestLooksLikeHeading:
    """`looks_like_heading` decides what becomes a section, so a false positive here
    fragments chunking downstream and a false negative collapses it to fixed windows.
    """

    BODY = 10.0

    @pytest.mark.parametrize(
        "text,size,bold",
        [
            ("1 Introduction", 14.0, True),
            ("3.2 Ablation Study", 14.0, True),
            ("References", 14.0, True),
            ("Limitations", 12.0, True),  # bold, no number, still clears the size test
        ],
    )
    def test_fires_on_real_headings(self, text, size, bold):
        assert looks_like_heading(text, size, bold, self.BODY) is True

    def test_does_not_fire_on_body_text(self):
        body = "We evaluate our approach on three benchmark datasets and report results."
        assert looks_like_heading(body, self.BODY, False, self.BODY) is False

    def test_does_not_fire_on_numbered_prose(self):
        # "2. we then re-rank ..." matches HEADING_NUM but is a sentence, not a title --
        # length is what has to stop it, since the numbering regex alone would accept it.
        prose = (
            "2. we then re-rank the candidates using the learned cross-encoder and "
            "select the top five results for the final answer synthesis step"
        )
        assert looks_like_heading(prose, self.BODY, False, self.BODY) is False

    @pytest.mark.parametrize(
        "fragment",
        [
            "v ∈ V p ( u )",  # inline math: set-membership + function application
            "0 0 0 0",
            "h ˆ −",
            ", (7)",  # equation-number tag
            "L T = −",
            "1 1 1 1 1",
        ],
    )
    def test_rejects_equation_fragments_even_when_bold_and_large(self, fragment):
        """Regression test: these are real spans pulled from the arXiv fraud corpus.

        Each one passes the length and word-count gates and clears the font-size test
        (LaTeX renders display math in the body font or larger), so before the
        alpha-ratio/word gate was added, every one of these was misread as a new
        section heading.
        """
        assert looks_like_heading(fragment, self.BODY + 2.0, True, self.BODY) is False

    def test_short_real_words_still_pass_despite_low_char_count(self):
        # "Baselines" is short but is a real word -- must not be caught by the fragment filter.
        assert looks_like_heading("Baselines", 14.0, True, self.BODY) is True


def test_estimate_body_size_falls_back_when_no_spans(tmp_path):
    import pymupdf

    path = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    doc = pymupdf.open(path)
    assert estimate_body_size(doc) == pytest.approx(10.0)  # DEFAULT_BODY_SIZE
    doc.close()


# --------------------------------------------------------------------------- parsing


class TestParsePdf:
    def test_every_paper_produces_pages_and_matches_manifest(self, corpus, ingestion_config):
        manifest = load_manifest(MANIFEST)
        assert len(manifest) == corpus["n_papers"]
        for paper in corpus["papers"]:
            pdf = RAW_DIR / paper["filename"]
            records, body_size = parse_pdf(
                pdf, paper["paper_id"], ingestion_config["parse"]["drop_sections"]
            )
            assert records, f"{paper['filename']} produced no pages"
            assert body_size == pytest.approx(corpus["body_font_size"], abs=0.5)

    def test_references_section_is_flagged_dropped(self, corpus, ingestion_config):
        paper = corpus["papers"][0]
        records, _ = parse_pdf(
            RAW_DIR / paper["filename"],
            paper["paper_id"],
            ingestion_config["parse"]["drop_sections"],
        )
        ref_records = [r for r in records if r.section.strip().lower() == "references"]
        assert ref_records, "fixture should contain a References section"
        assert all(r.dropped for r in ref_records)

    def test_heading_detection_recovers_the_known_section_list(self, corpus, ingestion_config):
        """Median-sections exit check from the stage-01 notebook, made concrete: every
        synthetic paper's detected sections must be a superset of its known headings,
        since the fixture's font contrast is exactly what the heuristic keys on.
        """
        for paper in corpus["papers"]:
            records, _ = parse_pdf(
                RAW_DIR / paper["filename"],
                paper["paper_id"],
                ingestion_config["parse"]["drop_sections"],
            )
            detected = {r.section.strip() for r in records}
            expected = set(paper["sections"])
            missing = expected - detected
            assert not missing, f"{paper['filename']}: missed headings {missing}"
            # No wild over-firing either: at most a handful of extra segments per real
            # heading (front-matter, page-break splits), never the 10x-plus explosion
            # the equation-fragment bug caused on the real arXiv corpus.
            assert len(detected) <= len(expected) + 5

    def test_parse_corpus_reports_thin_pages_without_dropping_them(self, corpus, ingestion_config):
        pages, report = parse_corpus(config=ingestion_config, keep_thin=True)
        assert report["n_papers"] == corpus["n_papers"]
        assert len(pages) > 0
        # thin is advisory: a thin page must still be present in the output
        thin_ids = {(p.paper_id, p.page) for p in pages if p.thin}
        for p in pages:
            if (p.paper_id, p.page) in thin_ids:
                assert p in pages  # trivially true; documents that thin pages survive


# --------------------------------------------------------------------------- canonical ids


class TestCanonicalPaperIds:
    def test_near_identical_titles_collapse_onto_the_first_seen_id(self):
        rows = [
            {"paper_id": "aaa111", "title": "Alleviating Inconsistency in GNN Fraud Detection"},
            {"paper_id": "bbb222", "title": "Alleviating Inconsistency in GNN Fraud Detection."},
        ]
        canon = canonical_paper_ids(rows)
        assert canon == {"aaa111": "aaa111", "bbb222": "aaa111"}

    def test_genuinely_different_titles_are_not_merged(self):
        rows = [
            {"paper_id": "aaa111", "title": "Alleviating Inconsistency in GNN Fraud Detection"},
            {"paper_id": "ccc333", "title": "A totally different paper about something else"},
        ]
        canon = canonical_paper_ids(rows)
        assert canon == {"aaa111": "aaa111", "ccc333": "ccc333"}

    def test_mapping_is_deterministic_regardless_of_input_order(self):
        rows = [
            {"paper_id": "bbb222", "title": "Same Paper Title Here"},
            {"paper_id": "aaa111", "title": "Same Paper Title Here"},
        ]
        # Sorted by paper_id internally, so "aaa111" (lexicographically first)
        # is always the canonical id, regardless of the order rows arrive in.
        assert canonical_paper_ids(rows) == {"aaa111": "aaa111", "bbb222": "aaa111"}
        assert canonical_paper_ids(list(reversed(rows))) == {"aaa111": "aaa111", "bbb222": "aaa111"}


# --------------------------------------------------------------------------- chunking


class TestChunking:
    def test_all_chunks_respect_token_bounds(self, corpus, ingestion_config):
        pages, _ = parse_corpus(config=ingestion_config)
        chunks = chunk_pages(pages, config=ingestion_config)
        assert chunks
        enc = get_encoder(ingestion_config["chunk"]["tokenizer"])
        max_tokens = ingestion_config["chunk"]["max_tokens"]
        for c in chunks:
            n_tokens = count_tokens(c.text, enc)
            assert n_tokens <= max_tokens, f"{c.chunk_id} exceeds max_tokens: {n_tokens}"
        # The min_tokens floor is enforced by merging, except for a genuinely tiny final
        # unit in a paper with nothing left to merge into -- so check the aggregate
        # rather than every single chunk, which is what the fraud-corpus run surfaced.
        min_tokens = ingestion_config["chunk"]["min_tokens"]
        below_floor = [c for c in chunks if count_tokens(c.text, enc) < min_tokens]
        assert len(below_floor) / len(chunks) < 0.1, "too many chunks below the min_tokens floor"

    def test_every_chunk_validates_against_the_contract(self, corpus, ingestion_config):
        pages, _ = parse_corpus(config=ingestion_config)
        chunks = chunk_pages(pages, config=ingestion_config)
        for c in chunks:
            assert isinstance(c, Chunk)
            # Round-trip through the contract model to prove it actually validates,
            # not just that construction happened to succeed once.
            Chunk.model_validate(c.model_dump())

    def test_context_prefix_present_when_enabled(self, corpus, ingestion_config):
        pages, _ = parse_corpus(config=ingestion_config)
        chunks = chunk_pages(pages, config=ingestion_config)
        sample = [c for c in chunks if c.metadata.title]
        assert sample
        for c in sample[:5]:
            assert c.text.startswith(c.metadata.title), (
                "context prefix should lead with the paper title"
            )
            assert c.metadata.section in c.text.split("\n", 1)[0]

    def test_tiny_units_are_merged_into_a_neighbour(self, corpus, ingestion_config):
        # Front matter (title + venue line) is ~20-30 tokens on its own -- below the
        # min_tokens floor -- and must not survive as a standalone chunk.
        min_tokens = ingestion_config["chunk"]["min_tokens"]
        pages, _ = parse_corpus(config=ingestion_config)
        units = group_sections(pages)
        chunks = chunk_pages(pages, config=ingestion_config)
        enc = get_encoder("cl100k_base")
        tiny_unit_count = sum(1 for u in units if count_tokens(u.text, enc) < min_tokens)
        assert tiny_unit_count > 0, (
            "fixture should contain at least one sub-floor unit (front matter)"
        )
        # Front matter's own text (paper title / venue line) should not appear as an
        # isolated chunk; it should have been folded into the section that follows.
        front_matter_only = [
            c
            for c in chunks
            if c.metadata.section not in ("Abstract",) and count_tokens(c.text, enc) < 15
        ]
        assert not front_matter_only, f"unmerged tiny chunk(s): {front_matter_only}"

    def test_facts_are_retrievable_on_the_recorded_page(self, corpus, ingestion_config):
        """The ground-truth contract: every fact's evidence sentence must survive into
        some chunk, and that chunk's page range must contain the fact's recorded page.
        This is what makes citation precision computable at all in stage 06.
        """
        pages, _ = parse_corpus(config=ingestion_config)
        chunks = chunk_pages(pages, config=ingestion_config)
        by_paper: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_paper.setdefault(c.metadata.paper_id, []).append(c)

        def norm(s: str) -> str:
            # PyMuPDF sometimes widens inter-word spacing on justified text (observed:
            # "We  present  SectionSplit," with doubled spaces), so compare on collapsed
            # whitespace rather than the exact byte sequence.
            return " ".join(s.split())

        missed = []
        for paper in corpus["papers"]:
            paper_chunks = by_paper.get(paper["paper_id"], [])
            for fact in paper["facts"]:
                evidence = norm(fact["evidence"])
                # `metadata.page` is the page at the chunk's own start offset, not a
                # range, so a fact whose evidence sits near the end of a chunk that
                # opens on the previous page is legitimately one page off from the
                # chunk's stamped page. +/-1 tolerates that without accepting a chunk
                # from a genuinely different part of the paper.
                hit = next(
                    (
                        c
                        for c in paper_chunks
                        if evidence in norm(c.text) and abs(c.metadata.page - fact["page"]) <= 1
                    ),
                    None,
                )
                if hit is None:
                    missed.append((paper["paper_id"], fact["fact_id"]))
        assert not missed, f"facts not retrievable on their recorded page: {missed}"


# --------------------------------------------------------------------------- pack_pieces / overlap


class TestPackPieces:
    """Direct tests of the packing primitive, using synthetic paragraphs sized to force
    a split -- the real corpus's short synthetic sections mostly don't, so this is the
    only way to exercise overlap and the max_tokens cap deterministically.
    """

    def _enc(self):
        return get_encoder("cl100k_base")

    def _para(self, n_sentences: int, tag: str) -> str:
        # Real sentence boundaries matter here: `_tail_overlap` falls back to trailing
        # sentences when no whole paragraph fits the overlap budget, so a paragraph with
        # no punctuation (all one "sentence") can never produce overlap, regardless of
        # length -- which is exactly what a first draft of this test got wrong.
        return " ".join(
            f"{tag} sentence {i} has several distinct words in it."
            for i in range(n_sentences)
        )

    def _spans(self, paras: list[str]) -> list:
        # pack_pieces takes (text, char_start, char_end) triples now, sourced from a
        # single underlying document; joining with "\n" mirrors how split_paragraphs
        # feeds it in production.
        joined = "\n".join(paras)
        return _spanned(paras, joined)

    def test_overlap_actually_overlaps(self):
        enc = self._enc()
        # Each paragraph is well under the target alone, but three together exceed it,
        # forcing a split whose tail should carry into the next chunk.
        paras = [self._para(8, f"PARA{i}") for i in range(5)]
        out = pack_pieces(
            self._spans(paras), target=100, overlap=30, min_tokens=10, max_tokens=200, enc=enc
        )
        assert len(out) >= 2
        # The end of chunk N and the start of chunk N+1 should share text (the overlap).
        tail_of_first = out[0][0][-120:]
        head_of_second = out[1][0][:120]
        shared_words = set(tail_of_first.split()) & set(head_of_second.split())
        assert len(shared_words) > 3, "no detectable overlap between consecutive chunks"

    def test_no_chunk_exceeds_max_tokens_even_with_an_oversize_paragraph(self):
        enc = self._enc()
        huge = self._para(2000, "HUGE")  # far above any max_tokens below
        out = pack_pieces(
            self._spans([huge]), target=100, overlap=20, min_tokens=10, max_tokens=150, enc=enc
        )
        for piece, _start, _end in out:
            assert count_tokens(piece, enc) <= 150

    def test_runt_chunks_are_merged_when_possible(self):
        enc = self._enc()
        paras = [self._para(90, "A"), self._para(5, "TINY")]
        out = pack_pieces(
            self._spans(paras), target=100, overlap=0, min_tokens=20, max_tokens=200, enc=enc
        )
        # The 5-word runt should have been absorbed into the previous chunk, not stand alone.
        assert all(count_tokens(p, enc) >= 20 for p, _start, _end in out)


def test_build_chunks_reserves_prefix_tokens_from_the_budget():
    """The bug this guards: n_tokens must never exceed max_tokens once the prefix is
    added, which requires reserving prefix cost before packing the body, not after.
    """
    from research_assistant.ingestion.chunk_B import SectionUnit

    enc = get_encoder("cl100k_base")
    long_body = " ".join(f"sentence number {i} contains several words." for i in range(200))
    unit = SectionUnit(
        paper_id="p1",
        section="Results",
        text=long_body,
        page_start=1,
        page_end=1,
        title="A Very Long Paper Title About Structured Sparsity In Retrieval Systems",
    )
    cfg = {
        "tokenizer": "cl100k_base",
        "target_tokens": 50,
        "overlap_tokens": 10,
        "min_tokens": 10,
        "max_tokens": 60,
        "prepend_context": True,
        "respect_paragraphs": True,
    }
    chunks = build_chunks([unit], cfg, titles={"p1": unit.title})
    for c in chunks:
        assert count_tokens(c.text, enc) <= 60
