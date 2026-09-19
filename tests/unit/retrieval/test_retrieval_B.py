"""Unit tests for the retrieval stage.

Everything here is built from the `Chunk` contract in-process. No fixture depends on
`data/`, on a downloaded embedding model, or on another thread's output, so these
tests fail only when this code is wrong.
"""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from research_assistant.contracts.retrieval_J import (
    Chunk,
    ChunkMetadata,
    RetrievalResponse,
    RetrievedChunk,
)
from research_assistant.retrieval.bm25_B import (
    ENGLISH_STOPWORDS,
    BM25Index,
    tokenize,
)
from research_assistant.retrieval.hybrid_B import (
    FusedCandidate,
    fuse,
    reciprocal_rank_fusion,
    weighted_sum_fusion,
)
from research_assistant.retrieval.service_B import RetrievalService
from research_assistant.retrieval.vector_store_B import (
    InMemoryVectorStore,
    VectorStore,
    VectorStoreLike,
)

# --------------------------------------------------------------------------
# Fixtures: a tiny hand-written corpus with known properties.
# --------------------------------------------------------------------------

# c1 is the only chunk containing the rare exact token "bleurt" -- the sparse arm's
# reason to exist. c2 paraphrases it without ever using the word.
CORPUS: list[dict[str, Any]] = [
    {
        "chunk_id": "c1",
        "paper_id": "p1",
        "title": "Metrics for Generation",
        "section": "Evaluation",
        "page": 4,
        "text": "We report BLEURT alongside ROUGE for every ablation in this table.",
    },
    {
        "chunk_id": "c2",
        "paper_id": "p1",
        "title": "Metrics for Generation",
        "section": "Evaluation",
        "page": 5,
        "text": "Learned automatic scoring correlates better with human judgement "
        "than surface overlap measures.",
    },
    {
        "chunk_id": "c3",
        "paper_id": "p1",
        "title": "Metrics for Generation",
        "section": "Method",
        "page": 2,
        "text": "The encoder is trained with a contrastive objective over sampled pairs.",
    },
    {
        "chunk_id": "c4",
        "paper_id": "p2",
        "title": "Retrieval Augmentation",
        "section": "Method",
        "page": 1,
        "text": "A sparse retriever and a dense retriever are fused by reciprocal rank fusion.",
    },
    {
        "chunk_id": "c5",
        "paper_id": "p2",
        "title": "Retrieval Augmentation",
        "section": "Evaluation",
        "page": 6,
        "text": "Recall at twenty caps everything downstream; the reranker cannot recover a miss.",
    },
]


def _make_chunk(row: dict[str, Any]) -> Chunk:
    row = dict(row)
    chunk_id = row.pop("chunk_id")
    text = row.pop("text")
    return Chunk(chunk_id=chunk_id, text=text, metadata=ChunkMetadata(**row))


@pytest.fixture
def chunks() -> list[Chunk]:
    return [_make_chunk(row) for row in CORPUS]


@pytest.fixture
def bm25(chunks: list[Chunk]) -> BM25Index:
    return BM25Index.build(chunks, k1=1.5, b=0.75, stopwords="english")


class FakeEmbedder:
    """Maps a query to a stored vector. Keeps the dense arm deterministic and offline."""

    def __init__(self, vectors: dict[str, Sequence[float]], default: Sequence[float]) -> None:
        self._vectors = vectors
        self._default = default

    def embed_query(self, query: str) -> Sequence[float]:
        return self._vectors.get(query, self._default)


class ScriptedVectorStore:
    """A `VectorStoreLike` that returns a fixed ranking, so fusion can be tested alone."""

    def __init__(self, chunks: Sequence[Chunk], ranking: Sequence[tuple[str, float]]) -> None:
        self._by_id = {c.chunk_id: c for c in chunks}
        self._ranking = list(ranking)

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        for c in chunks:
            self._by_id[c.chunk_id] = c

    def search(self, embedding: Sequence[float], n: int) -> list[tuple[str, float]]:
        return self._ranking[:n]

    def count(self) -> int:
        return len(self._by_id)

    def get(self, chunk_id: str) -> Chunk:
        return self._by_id[chunk_id]

    def delete_collection(self) -> None:
        self._by_id.clear()


class IdentityRanker:
    """A ranker that preserves fusion order. Stands in for the registry's baseline."""

    name = "identity-test"

    def rank(self, query: str, chunks: Sequence[Chunk]) -> list[tuple[Chunk, float]]:
        return [(c, 1.0 / (i + 1)) for i, c in enumerate(chunks)]


class ReversingRanker:
    """Proves the ranker really decides final order, not fusion."""

    name = "reversing-test"

    def rank(self, query: str, chunks: Sequence[Chunk]) -> list[tuple[Chunk, float]]:
        return [(c, float(i)) for i, c in enumerate(reversed(list(chunks)))]


TEST_CONFIG: dict[str, Any] = {
    "hybrid": {
        "fusion": "rrf",
        "rrf_k": 60,
        "candidates_per_arm": 30,
        "candidate_k": 20,
        "weights": {"bm25": 0.4, "vector": 0.6},
        "dedupe_by": "chunk_id",
    },
    "rerank": {"active": "baseline", "top_k": 5},
}


def make_service(
    chunks: list[Chunk],
    bm25: BM25Index,
    *,
    vector_ranking: Sequence[tuple[str, float]],
    ranker: Any = None,
    config: dict[str, Any] | None = None,
) -> RetrievalService:
    cfg = config if config is not None else {k: dict(v) for k, v in TEST_CONFIG.items()}
    return RetrievalService(
        vector_store=ScriptedVectorStore(chunks, vector_ranking),
        bm25_index=bm25,
        embedder=FakeEmbedder({}, [0.0, 1.0]),
        ranker=ranker if ranker is not None else IdentityRanker(),
        config=cfg,
    )


# --------------------------------------------------------------------------
# Tokenisation and stopwords
# --------------------------------------------------------------------------


def test_tokenize_lowercases_and_splits_on_punctuation() -> None:
    assert tokenize("BLEURT, ROUGE-L; and BLEU!") == ["bleurt", "rouge", "l", "bleu"]


def test_tokenize_removes_english_stopwords() -> None:
    tokens = tokenize("the encoder is trained with a contrastive objective")
    assert tokens == ["encoder", "trained", "contrastive", "objective"]
    for stopword in ("the", "is", "with", "a"):
        assert stopword in ENGLISH_STOPWORDS
        assert stopword not in tokens


def test_tokenize_can_keep_stopwords() -> None:
    assert tokenize("the encoder", remove_stopwords=False) == ["the", "encoder"]


def test_tokenize_keeps_digits_inside_identifiers() -> None:
    # bge-small-en-v1.5 and f1 must survive as searchable units; these are exactly the
    # terms the dense arm loses.
    assert tokenize("bge-small-en-v1.5 improves F1") == [
        "bge",
        "small",
        "en",
        "v1",
        "5",
        "improves",
        "f1",
    ]


def test_stopword_only_query_returns_nothing_rather_than_everything(bm25: BM25Index) -> None:
    assert bm25.search("the of and with", 10) == []


# --------------------------------------------------------------------------
# BM25: the exact-term case that justifies the sparse arm
# --------------------------------------------------------------------------


def test_bm25_finds_exact_rare_term(bm25: BM25Index) -> None:
    assert bm25.search("BLEURT", 5)[0] == "c1"


def test_bm25_drops_zero_scoring_documents(bm25: BM25Index) -> None:
    # Only c1 contains "bleurt". Asking for 30 must not return all five chunks.
    assert bm25.search("BLEURT", 30) == ["c1"]


def test_paraphrase_query_misses_the_exact_term_chunk(bm25: BM25Index) -> None:
    """The dense arm's job, stated as a sparse-arm failure.

    A paraphrase that never says "BLEURT" cannot retrieve c1 lexically, which is why
    hybrid exists: the vector arm has to be the one that finds it.
    """
    paraphrase = "learned automatic scoring correlating with human judgement"
    hits = bm25.search(paraphrase, 5)
    assert "c1" not in hits
    assert hits and hits[0] == "c2"


def test_bm25_roundtrips_through_pickle(bm25: BM25Index, tmp_path: Any) -> None:
    path = tmp_path / "bm25.pkl"
    bm25.save(path)
    reloaded = BM25Index.load(path)
    assert len(reloaded) == len(bm25)
    assert reloaded.search("BLEURT", 5) == bm25.search("BLEURT", 5)
    assert reloaded.get("c1").text == bm25.get("c1").text


def test_bm25_load_rejects_a_foreign_pickle(tmp_path: Any) -> None:
    import pickle

    path = tmp_path / "old.pkl"
    path.write_bytes(pickle.dumps({"format": 1, "chunks": []}))
    with pytest.raises(ValueError, match="rebuild it"):
        BM25Index.load(path)


# --------------------------------------------------------------------------
# Reciprocal rank fusion: hand-computed arithmetic
# --------------------------------------------------------------------------


def test_rrf_scores_match_hand_computation() -> None:
    fused = reciprocal_rank_fusion(["a", "b"], ["b", "c"], k=60)
    scores = {c.chunk_id: c.score for c in fused}

    # a: bm25 rank 1 only            -> 1/61
    # b: bm25 rank 2, vector rank 1  -> 1/62 + 1/61
    # c: vector rank 2 only          -> 1/62
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)


def test_rrf_k_damps_the_head_of_each_list() -> None:
    small_k = {c.chunk_id: c.score for c in reciprocal_rank_fusion(["a"], ["b"], k=1)}
    large_k = {c.chunk_id: c.score for c in reciprocal_rank_fusion(["a"], ["b"], k=100)}
    assert small_k["a"] == pytest.approx(0.5)
    assert large_k["a"] == pytest.approx(1 / 101)


def test_chunk_in_both_arms_beats_chunk_ranked_first_by_one_arm() -> None:
    """The central claim of hybrid retrieval, as arithmetic.

    `both` is second in each arm; `solo` is first in the sparse arm and absent from
    the dense one. RRF must still prefer `both`: 1/62 + 1/62 > 1/61.
    """
    fused = reciprocal_rank_fusion(["solo", "both"], ["other", "both"], k=60)
    order = [c.chunk_id for c in fused]
    assert order[0] == "both"
    assert order.index("both") < order.index("solo")
    assert fused[0].score == pytest.approx(2 / 62)


def test_rrf_keeps_per_arm_ranks_for_the_trace() -> None:
    fused = {c.chunk_id: c for c in reciprocal_rank_fusion(["a", "b"], ["b", "c"], k=60)}

    assert (fused["a"].bm25_rank, fused["a"].vector_rank) == (1, None)
    assert (fused["b"].bm25_rank, fused["b"].vector_rank) == (2, 1)
    assert (fused["c"].bm25_rank, fused["c"].vector_rank) == (None, 2)

    assert fused["b"].arms == ("bm25", "vector")
    assert fused["a"].arms == ("bm25",)
    assert fused["c"].arms == ("vector",)


def test_rrf_dedupes_by_chunk_id() -> None:
    """A chunk found by both arms appears once, not twice."""
    fused = reciprocal_rank_fusion(["x", "y"], ["x", "y"], k=60)
    ids = [c.chunk_id for c in fused]
    assert ids == sorted(set(ids), key=ids.index)
    assert len(ids) == 2


def test_rrf_dedupes_repeats_within_one_arm() -> None:
    """A duplicate id inside one arm keeps its best rank and is not double-counted."""
    fused = reciprocal_rank_fusion(["x", "x", "y"], [], k=60)
    by_id = {c.chunk_id: c for c in fused}
    assert len(fused) == 2
    assert by_id["x"].bm25_rank == 1
    assert by_id["x"].score == pytest.approx(1 / 61)


def test_rrf_ordering_is_deterministic_under_ties() -> None:
    a = [c.chunk_id for c in reciprocal_rank_fusion(["q"], ["z"], k=60)]
    b = [c.chunk_id for c in reciprocal_rank_fusion(["q"], ["z"], k=60)]
    assert a == b == ["q", "z"]


# --------------------------------------------------------------------------
# Weighted sum: the configured alternative
# --------------------------------------------------------------------------


def test_weighted_sum_normalises_each_arm_independently() -> None:
    # BM25 scores on an unbounded scale, vector distances in [0, 2]: the two must be
    # min-max normalised before the weights mean anything.
    fused = {
        c.chunk_id: c.score
        for c in weighted_sum_fusion(
            [("a", 40.0), ("b", 20.0)],
            [("b", 0.10), ("c", 0.50)],
            bm25_weight=0.4,
            vector_weight=0.6,
        )
    }
    assert fused["a"] == pytest.approx(0.4 * 1.0)
    assert fused["b"] == pytest.approx(0.4 * 0.0 + 0.6 * 1.0)
    assert fused["c"] == pytest.approx(0.6 * 0.0)


def test_weighted_sum_treats_smaller_distance_as_better() -> None:
    fused = weighted_sum_fusion([], [("near", 0.05), ("far", 0.95)], vector_weight=1.0)
    assert [c.chunk_id for c in fused] == ["near", "far"]


def test_weighted_sum_with_a_flat_arm_keeps_that_arms_weight() -> None:
    # All-equal scores must not collapse to zero and silently delete the arm.
    fused = {
        c.chunk_id: c.score
        for c in weighted_sum_fusion(
            [("a", 5.0), ("b", 5.0)], [], bm25_weight=0.4, vector_weight=0.6
        )
    }
    assert fused["a"] == pytest.approx(0.4)
    assert fused["b"] == pytest.approx(0.4)


def test_weighted_sum_keeps_per_arm_ranks() -> None:
    fused = {c.chunk_id: c for c in weighted_sum_fusion([("a", 9.0)], [("a", 0.1), ("b", 0.2)])}
    assert (fused["a"].bm25_rank, fused["a"].vector_rank) == (1, 1)
    assert (fused["b"].bm25_rank, fused["b"].vector_rank) == (None, 2)


def test_fuse_dispatches_on_the_config_key() -> None:
    # "a" is a near-tie winner in the sparse arm and a distant second in the dense arm.
    args = ([("a", 100.0), ("b", 99.0)], [("c", 0.1), ("a", 0.9)])
    kwargs: dict[str, Any] = {"rrf_k": 60, "weights": {"bm25": 0.4, "vector": 0.6}}

    rrf_ids = [c.chunk_id for c in fuse(*args, fusion="rrf", candidate_k=10, **kwargs)]
    ws_ids = [c.chunk_id for c in fuse(*args, fusion="weighted_sum", candidate_k=10, **kwargs)]

    assert set(rrf_ids) == set(ws_ids) == {"a", "b", "c"}
    # RRF sees only ranks, so "a" wins for appearing in both arms. Weighted sum sees
    # raw margins: "a" normalises to 0 on the dense arm despite ranking second, and
    # "c" takes the head. The two strategies genuinely disagree, which is what makes
    # the `hybrid.fusion` config key load-bearing rather than cosmetic.
    assert rrf_ids[0] == "a"
    assert ws_ids[0] == "c"


def test_fuse_rejects_an_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown hybrid.fusion"):
        fuse([], [], fusion="learned", rrf_k=60, weights={}, candidate_k=5)


def test_fuse_truncates_to_candidate_k() -> None:
    bm25 = [(f"c{i}", float(50 - i)) for i in range(30)]
    vector = [(f"v{i}", float(i) / 100) for i in range(30)]
    assert len(fuse(bm25, vector, fusion="rrf", rrf_k=60, weights={}, candidate_k=20)) == 20


# --------------------------------------------------------------------------
# Service: candidate_k, top_k, filters, contract validity
# --------------------------------------------------------------------------


def test_search_returns_a_valid_retrieval_response(chunks: list[Chunk], bm25: BM25Index) -> None:
    service = make_service(chunks, bm25, vector_ranking=[("c2", 0.1), ("c1", 0.3)])
    result = service.search("BLEURT scores")

    assert isinstance(result, RetrievalResponse)
    RetrievalResponse.model_validate(result.model_dump())  # round-trips as the tools will
    assert result.query == "BLEURT scores"
    assert result.reranker_version == "identity-test"
    assert result.latency_ms >= 0.0
    assert all(isinstance(r, RetrievedChunk) for r in result.results)
    assert [r.rank for r in result.results] == list(range(1, len(result.results) + 1))


def test_search_populates_per_arm_scores_so_a_trace_names_the_arm(
    chunks: list[Chunk], bm25: BM25Index
) -> None:
    service = make_service(chunks, bm25, vector_ranking=[("c3", 0.1), ("c1", 0.2)])
    result = service.search("BLEURT")

    by_id = {r.chunk.chunk_id: r for r in result.results}
    assert by_id["c1"].bm25_score is not None  # the sparse arm found the exact term
    assert by_id["c1"].vector_score is not None
    assert by_id["c3"].bm25_score is None  # dense-only hit
    assert by_id["c3"].vector_score is not None


def test_search_honours_top_k(chunks: list[Chunk], bm25: BM25Index) -> None:
    ranking = [(c.chunk_id, i / 10) for i, c in enumerate(chunks)]
    service = make_service(chunks, bm25, vector_ranking=ranking)

    assert len(service.search("retriever fusion", top_k=2).results) == 2
    assert len(service.search("retriever fusion", top_k=5).results) == 5
    # Config default is 5.
    assert len(service.search("retriever fusion").results) == 5


def test_search_honours_candidate_k(chunks: list[Chunk], bm25: BM25Index) -> None:
    # `candidate_count` isn't on the merged RetrievalResponse contract, so the cap is
    # observed through `results` length with `top_k` set >= `candidate_k` so it never
    # masks the effect.
    ranking = [(c.chunk_id, i / 10) for i, c in enumerate(chunks)]
    service = make_service(chunks, bm25, vector_ranking=ranking)

    assert len(service.search("retriever fusion", candidate_k=2, top_k=5).results) == 2
    assert len(service.search("retriever fusion", candidate_k=5, top_k=5).results) == 5


def test_candidate_k_caps_recall_permanently(chunks: list[Chunk], bm25: BM25Index) -> None:
    """Anything cut at candidate_k cannot be recovered by the reranker.

    With candidate_k=1 the relevant chunk is outside the candidate set, so even a
    ranker that would have placed it first never sees it. This is why recall@20 is a
    gate threshold rather than a report footnote.
    """
    service = make_service(
        chunks, bm25, vector_ranking=[("c3", 0.1), ("c4", 0.2), ("c1", 0.3)]
    )
    wide = service.search("contrastive objective encoder", candidate_k=5, top_k=5)
    narrow = service.search("contrastive objective encoder", candidate_k=1, top_k=5)

    assert len(narrow.results) == 1
    assert len(wide.results) > len(narrow.results)
    assert set(r.chunk.chunk_id for r in narrow.results) < set(
        r.chunk.chunk_id for r in wide.results
    )


def test_search_dedupes_a_chunk_found_by_both_arms(chunks: list[Chunk], bm25: BM25Index) -> None:
    service = make_service(chunks, bm25, vector_ranking=[("c1", 0.05), ("c2", 0.2)])
    result = service.search("BLEURT")

    ids = [r.chunk.chunk_id for r in result.results]
    assert ids.count("c1") == 1
    assert len(ids) == len(set(ids))


def test_search_filters_by_paper_id(chunks: list[Chunk], bm25: BM25Index) -> None:
    ranking = [(c.chunk_id, i / 10) for i, c in enumerate(chunks)]
    service = make_service(chunks, bm25, vector_ranking=ranking)

    result = service.search("retriever fusion evaluation", filters={"paper_id": "p2"})
    assert result.results
    assert {r.chunk.metadata.paper_id for r in result.results} == {"p2"}


def test_search_filters_by_section_case_insensitively(
    chunks: list[Chunk], bm25: BM25Index
) -> None:
    ranking = [(c.chunk_id, i / 10) for i, c in enumerate(chunks)]
    service = make_service(chunks, bm25, vector_ranking=ranking)

    result = service.search("retriever fusion evaluation", filters={"section": "method"})
    assert result.results
    assert {r.chunk.metadata.section for r in result.results} == {"Method"}


def test_search_ignores_none_valued_filters(chunks: list[Chunk], bm25: BM25Index) -> None:
    # SearchPapersInput sends paper_id=None / section=None when unset.
    ranking = [(c.chunk_id, i / 10) for i, c in enumerate(chunks)]
    service = make_service(chunks, bm25, vector_ranking=ranking)

    unfiltered = service.search("retriever fusion evaluation")
    passthrough = service.search(
        "retriever fusion evaluation", filters={"paper_id": None, "section": None}
    )
    assert [r.chunk.chunk_id for r in passthrough.results] == [
        r.chunk.chunk_id for r in unfiltered.results
    ]


def test_the_ranker_not_fusion_decides_final_order(chunks: list[Chunk], bm25: BM25Index) -> None:
    ranking = [(c.chunk_id, i / 10) for i, c in enumerate(chunks)]
    forward = make_service(chunks, bm25, vector_ranking=ranking, ranker=IdentityRanker())
    reverse = make_service(chunks, bm25, vector_ranking=ranking, ranker=ReversingRanker())

    a = [r.chunk.chunk_id for r in forward.search("retriever", top_k=5).results]
    b = [r.chunk.chunk_id for r in reverse.search("retriever", top_k=5).results]
    assert a == list(reversed(b))


def test_result_names_the_ranker_that_produced_it(chunks: list[Chunk], bm25: BM25Index) -> None:
    service = make_service(chunks, bm25, vector_ranking=[("c1", 0.1)], ranker=ReversingRanker())
    assert service.search("BLEURT").reranker_version == "reversing-test"


def test_weighted_sum_config_changes_the_result(chunks: list[Chunk], bm25: BM25Index) -> None:
    # `fusion` isn't stamped on the merged RetrievalResponse contract, so the config
    # switch is observed through a changed ranking rather than an echoed field.
    rrf_cfg = {k: dict(v) for k, v in TEST_CONFIG.items()}
    ws_cfg = {k: dict(v) for k, v in TEST_CONFIG.items()}
    ws_cfg["hybrid"]["fusion"] = "weighted_sum"

    ranking = [("c2", 0.05), ("c1", 0.9)]
    rrf_result = make_service(chunks, bm25, vector_ranking=ranking, config=rrf_cfg).search(
        "BLEURT"
    )
    ws_result = make_service(chunks, bm25, vector_ranking=ranking, config=ws_cfg).search("BLEURT")

    RetrievalResponse.model_validate(rrf_result.model_dump())
    RetrievalResponse.model_validate(ws_result.model_dump())
    rrf_order = [r.chunk.chunk_id for r in rrf_result.results]
    ws_order = [r.chunk.chunk_id for r in ws_result.results]
    assert rrf_order != ws_order


def test_search_with_no_hits_is_still_a_valid_result(chunks: list[Chunk], bm25: BM25Index) -> None:
    service = make_service(chunks, bm25, vector_ranking=[])
    result = service.search("zzzz nonexistent token")

    assert result.results == []
    RetrievalResponse.model_validate(result.model_dump())


def test_hybrid_beats_either_arm_alone_on_the_corpus(
    chunks: list[Chunk], bm25: BM25Index
) -> None:
    """The ablation in miniature.

    A query mixing an exact identifier with a paraphrase: the sparse arm finds only
    c1, the dense arm only c2, and fusion returns both.
    """
    sparse_only = bm25.search("BLEURT", 30)
    dense_only = [cid for cid, _ in [("c2", 0.1), ("c5", 0.4)]]
    fused = [c.chunk_id for c in reciprocal_rank_fusion(sparse_only, dense_only, k=60)]

    assert "c1" in sparse_only and "c1" not in dense_only
    assert "c2" in dense_only and "c2" not in sparse_only
    assert {"c1", "c2"} <= set(fused)


# --------------------------------------------------------------------------
# Service helpers used by Sude's get_citation / summarize_section
# --------------------------------------------------------------------------


def test_get_chunk_returns_the_contract_type(chunks: list[Chunk], bm25: BM25Index) -> None:
    service = make_service(chunks, bm25, vector_ranking=[])
    chunk = service.get_chunk("c1")
    assert isinstance(chunk, Chunk)
    assert chunk.metadata.paper_id == "p1"
    assert chunk.metadata.page == 4


def test_get_chunk_returns_none_for_an_unknown_id(chunks: list[Chunk], bm25: BM25Index) -> None:
    service = make_service(chunks, bm25, vector_ranking=[])
    assert service.get_chunk("does-not-exist") is None


def test_get_section_returns_one_papers_section_in_reading_order(
    chunks: list[Chunk], bm25: BM25Index
) -> None:
    service = make_service(chunks, bm25, vector_ranking=[])
    section = service.get_section("p1", "Evaluation")

    assert [c.chunk_id for c in section] == ["c1", "c2"]
    assert [c.metadata.page for c in section] == [4, 5]
    # p2 also has an "Evaluation" section; it must not leak in.
    assert {c.metadata.paper_id for c in section} == {"p1"}


def test_get_section_matches_case_insensitively(chunks: list[Chunk], bm25: BM25Index) -> None:
    service = make_service(chunks, bm25, vector_ranking=[])
    assert service.get_section("p1", "evaluation") == service.get_section("p1", "Evaluation")


def test_get_section_returns_empty_for_an_unknown_section(
    chunks: list[Chunk], bm25: BM25Index
) -> None:
    service = make_service(chunks, bm25, vector_ranking=[])
    assert service.get_section("p1", "Conclusion") == []


# --------------------------------------------------------------------------
# The arms are independently testable and independently timed
# --------------------------------------------------------------------------


def test_arms_can_be_called_separately(chunks: list[Chunk], bm25: BM25Index) -> None:
    service = make_service(chunks, bm25, vector_ranking=[("c3", 0.1), ("c4", 0.2)])

    sparse = service.bm25_arm("BLEURT", 30)
    dense = service.vector_arm("BLEURT", 30)

    assert sparse == [("c1", pytest.approx(sparse[0][1]))]
    assert [cid for cid, _ in dense] == ["c3", "c4"]


def test_in_memory_store_satisfies_the_protocol() -> None:
    assert isinstance(InMemoryVectorStore(), VectorStoreLike)


# --------------------------------------------------------------------------
# Contract guards
# --------------------------------------------------------------------------


def test_retrieved_chunk_rejects_rank_zero(chunks: list[Chunk]) -> None:
    with pytest.raises(ValidationError):
        RetrievedChunk(chunk=chunks[0], score=1.0, rank=0)


# NOTE: the merged `retrieval_J.RetrievalResponse` contract dropped both the
# `fusion` field and any lower bound on `ChunkMetadata.page` (it is now an
# unconstrained `int | None`), so the two guards this file used to pin
# (`test_retrieval_result_rejects_an_unknown_fusion`, `test_chunk_rejects_page_zero`)
# no longer have anything to assert against. Flagged for Sude rather than silently
# dropped: a page of 0 or negative is accepted by the schema today.


# --------------------------------------------------------------------------
# Integration-style: a real Chroma collection, built and torn down in a temp dir
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_dims", [4])
def test_real_chroma_roundtrip(chunks: list[Chunk], tmp_path: Any, n_dims: int) -> None:
    """Proves the Chroma mapping really rebuilds a Chunk, metadata and all.

    The notebook prototype persisted only five metadata keys, which would have made
    `get()` impossible; this test is what pins the fix.
    """
    chromadb = pytest.importorskip("chromadb")
    assert chromadb  # keep the import meaningful to linters

    persist_dir = tmp_path / "chroma"
    store = VectorStore(
        persist_dir=persist_dir, collection="test_papers_v1", distance="cosine"
    )
    try:
        vectors = [
            [1.0 if i == j % n_dims else 0.0 for i in range(n_dims)]
            for j, _ in enumerate(chunks)
        ]
        store.upsert(chunks, vectors)
        assert store.count() == len(chunks)

        restored = store.get("c5")
        assert restored == chunks[4]  # every field survives the round trip

        hits = store.search(vectors[0], 3)
        assert hits and hits[0][0] == "c1"
        assert all(isinstance(cid, str) and isinstance(d, float) for cid, d in hits)

        with pytest.raises(KeyError):
            store.get("not-a-chunk")

        store.delete_collection()
        assert store.count() == 0
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)


def test_upsert_rejects_mismatched_lengths(chunks: list[Chunk]) -> None:
    store = InMemoryVectorStore()
    with pytest.raises(ValueError, match="length mismatch"):
        store.upsert(chunks, [[0.0]])


def test_fused_candidate_is_hashable_and_frozen() -> None:
    candidate = FusedCandidate(chunk_id="a", score=0.5, bm25_rank=1)
    assert hash(candidate)
    with pytest.raises(dataclasses.FrozenInstanceError):
        candidate.score = 0.9  # type: ignore[misc]
