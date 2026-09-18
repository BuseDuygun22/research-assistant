"""Judge tests (Sude).

The rubric is the unit under test here, not the model. `derive_passed` is a pure
function over counted violations, which is the whole reason `passed` is computed
rather than self-reported — it can be tested without a model in the loop.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from research_assistant.contracts.judge_J import FaithfulnessViolation, JudgeMeta
from research_assistant.contracts.retrieval_J import Chunk, ChunkMetadata, RetrievedChunk
from research_assistant.judge.faithfulness_S import (
    MIN_CITATION_PRECISION,
    MIN_COVERAGE,
    PanelMember,
    derive_passed,
    panel_faithfulness,
)
from research_assistant.judge.relevance_S import build_preference_pairs, order_seed
from research_assistant.llm_S import set_llm

META = JudgeMeta(judge_model="stub", prompt_version="v1")


def viol(kind: str, severity: str = "major") -> FaithfulnessViolation:
    return FaithfulnessViolation(
        kind=kind,  # type: ignore[arg-type]
        claim=f"claim-{kind}",
        explanation="x",
        severity=severity,  # type: ignore[arg-type]
    )


def chunk(cid: str, text: str = "text") -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=text,
        metadata=ChunkMetadata(paper_id="p1", title="A Paper"),
    )


def retrieved(cid: str) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk(cid), score=1.0, rank=1)


@pytest.fixture(autouse=True)
def reset():
    yield
    set_llm(None)


# --- the pass rule -----------------------------------------------------------


def test_clean_draft_passes():
    assert derive_passed([], 1.0, 1.0) is True


@pytest.mark.parametrize("kind", ["unsupported_claim", "inferential_leap", "contradicts_source"])
def test_one_deep_violation_fails_regardless_of_ratios(kind):
    """A cited report containing one fabricated claim is worse than no report:
    the citation lends the fabrication unearned credibility."""
    assert derive_passed([viol(kind)], 1.0, 1.0) is False


def test_major_shallow_violation_fails():
    assert derive_passed([viol("missing_citation")], 1.0, 1.0) is False


def test_minor_shallow_violation_does_not_block():
    assert derive_passed([viol("missing_citation", "minor")], 1.0, 1.0) is True


def test_precision_below_threshold_fails():
    assert derive_passed([], MIN_CITATION_PRECISION - 0.01, 1.0) is False


def test_coverage_below_threshold_fails():
    assert derive_passed([], 1.0, MIN_COVERAGE - 0.01) is False


def test_coverage_threshold_is_not_one():
    """Demanding a citation on every sentence produces citation-stuffing, not
    grounding — transitions and hedges are excluded from claims."""
    assert MIN_COVERAGE < 1.0


# --- ordering ----------------------------------------------------------------


def test_order_seed_is_stable_for_a_query():
    assert order_seed("what is RRF") == order_seed("what is RRF")


def test_order_seed_differs_across_queries():
    assert order_seed("what is RRF") != order_seed("what is DPR")


# --- preference pairs --------------------------------------------------------


def test_pairs_are_built_only_above_the_margin():
    from research_assistant.contracts.judge_J import RelevanceVerdict

    verdicts = [
        RelevanceVerdict(
            query="q", chunk_id="a", grade=3, confidence=0.9, rationale="", meta=META
        ),
        RelevanceVerdict(
            query="q", chunk_id="b", grade=2, confidence=0.9, rationale="", meta=META
        ),
        RelevanceVerdict(
            query="q", chunk_id="c", grade=0, confidence=0.9, rationale="", meta=META
        ),
    ]
    chunks = [chunk("a"), chunk("b"), chunk("c")]

    loose = build_preference_pairs("q", verdicts, chunks, min_margin=1.0)
    strict = build_preference_pairs("q", verdicts, chunks, min_margin=2.0)
    assert len(strict) < len(loose)
    assert all(p.margin >= 2.0 for p in strict)


def test_every_pair_retains_its_margin():
    """Retained rather than pre-filtered so Buse can raise the threshold without
    regenerating the dataset."""
    from research_assistant.contracts.judge_J import RelevanceVerdict

    verdicts = [
        RelevanceVerdict(
            query="q", chunk_id="a", grade=3, confidence=0.9, rationale="", meta=META
        ),
        RelevanceVerdict(
            query="q", chunk_id="b", grade=1, confidence=0.9, rationale="", meta=META
        ),
    ]
    pairs = build_preference_pairs("q", verdicts, [chunk("a"), chunk("b")])
    assert pairs and all(p.margin > 0 for p in pairs)


# --- the panel ---------------------------------------------------------------


class FixedLLM:
    """A judge that always reports the same findings."""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        return ""

    def complete_json(
        self, system: str, user: str, schema: type[BaseModel], max_tokens: int = 1024
    ) -> BaseModel:
        return schema.model_validate(self.payload)


CLEAN = {
    "violations": [],
    "total_claims": 4,
    "cited_claims": 4,
    "supported_cited_claims": 4,
    "confidence": 0.9,
}
DIRTY = {
    "violations": [viol("unsupported_claim").model_dump()],
    "total_claims": 4,
    "cited_claims": 4,
    "supported_cited_claims": 4,
    "confidence": 0.9,
}


def test_panel_needs_members():
    with pytest.raises(ValueError):
        panel_faithfulness("d", "draft", [retrieved("a")], [])


def test_unanimous_panel_keeps_full_confidence():
    members = [PanelMember(f"m{i}", FixedLLM(CLEAN)) for i in range(3)]
    v = panel_faithfulness("d", "draft", [retrieved("a")], members)
    assert v.passed is True
    assert v.confidence == pytest.approx(0.9)


def test_split_panel_loses_confidence():
    """The property worth having: disagreement between judges becomes a reason to
    involve a human, instead of being averaged away."""
    members = [
        PanelMember("a", FixedLLM(CLEAN)),
        PanelMember("b", FixedLLM(DIRTY)),
        PanelMember("c", FixedLLM(CLEAN)),
    ]
    v = panel_faithfulness("d", "draft", [retrieved("a")], members)
    assert v.passed is True  # 2-1 majority
    assert v.confidence < 0.9  # but the panel is less sure than any member


def test_panel_majority_not_unanimity():
    """One stubborn member must not be able to veto."""
    members = [
        PanelMember("a", FixedLLM(DIRTY)),
        PanelMember("b", FixedLLM(DIRTY)),
        PanelMember("c", FixedLLM(CLEAN)),
    ]
    assert panel_faithfulness("d", "draft", [retrieved("a")], members).passed is False


def test_panel_unions_violations():
    """Recall matters more than precision here: a missed deep violation ships, a
    spurious one costs a revision round."""
    members = [PanelMember("a", FixedLLM(CLEAN)), PanelMember("b", FixedLLM(DIRTY))]
    v = panel_faithfulness("d", "draft", [retrieved("a")], members)
    assert len(v.violations) == 1


def test_panel_stamps_its_members():
    members = [PanelMember("a", FixedLLM(CLEAN)), PanelMember("b", FixedLLM(CLEAN))]
    v = panel_faithfulness("d", "draft", [retrieved("a")], members)
    assert v.meta.judge_model.startswith("panel:")
