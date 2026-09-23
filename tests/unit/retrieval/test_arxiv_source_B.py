"""arXiv query construction (Sude/Buse shared).

Regression coverage for a real bug found running live discovery for real: a
multi-word term was auto-wrapped in exact-phrase quotes, so a full
reformulated question like "federated learning cross-bank fraud detection
privacy-preserving mechanisms" was searched as one literal 8-word phrase -
which essentially never appears verbatim in any abstract, so arXiv correctly
returned zero results every single time, and live discovery could never find
anything regardless of how good the query was.
"""

from __future__ import annotations

from research_assistant.retrieval.arxiv_source_B import DEFAULT_CATEGORIES, build_query


def test_a_multi_word_term_is_not_auto_phrase_quoted():
    """The bug, pinned: this must be an AND-of-words search, not one literal
    phrase - no %22 anywhere in the query."""
    q = build_query("federated learning fraud detection", DEFAULT_CATEGORIES)
    assert "%22" not in q
    assert "all:federated+learning+fraud+detection" in q


def test_a_caller_that_wants_an_exact_phrase_pre_encodes_it_itself():
    """`fetch_arxiv_fraud_B.py` wants "fraud detection" as one phrase, and gets
    it by pre-encoding %22...%22 before calling - `build_query` must pass an
    already-quoted term through unchanged rather than re-encoding it."""
    q = build_query("%22fraud+detection%22", DEFAULT_CATEGORIES)
    assert "all:%22fraud+detection%22" in q


def test_categories_are_anded_as_an_or_group():
    q = build_query("x", ["cs.LG", "cs.CR"])
    assert "%28cat:cs.LG+OR+cat:cs.CR%29" in q
    assert "+AND+" in q


def test_single_word_term_is_unaffected():
    assert "all:fraud" in build_query("fraud", DEFAULT_CATEGORIES)
