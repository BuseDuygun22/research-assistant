"""Unit tests for the reranker package — Track A / Buse.

The leakage test is the one that must never be flaky: it uses hand-built data, no
filesystem beyond tmp_path, no network, no model. If it ever goes red, stop and
look at the data, not at the test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_assistant.contracts.judge_J import JudgeMeta, RelevanceVerdict
from research_assistant.contracts.retrieval_J import Chunk, ChunkMetadata
from research_assistant.reranker import pairs_B, registry_B, train_B
from research_assistant.reranker.baseline_B import BaselineRanker, Ranker

# --------------------------------------------------------------------------- #
# fixtures / builders
# --------------------------------------------------------------------------- #


def make_score(query: str, chunk_id: str, grade: int) -> RelevanceVerdict:
    return RelevanceVerdict(
        query=query,
        chunk_id=chunk_id,
        grade=grade,  # type: ignore[arg-type]
        confidence=1.0,
        rationale="because",
        meta=JudgeMeta(judge_model="test-judge", prompt_version="v1"),
    )


def lookup(chunk_id: str) -> str:
    """Stand-in `ChunkLookup`: `RelevanceVerdict` carries no chunk text, so tests
    that don't care about text content resolve it deterministically from the id."""
    return f"text of {chunk_id}"


def make_chunk(chunk_id: str, text: str = "some text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(paper_id="p1", title="A paper", section="Methods", page=1),
    )


@pytest.fixture
def tmp_registry(tmp_path: Path) -> Path:
    return tmp_path / "registry_B.json"


# --------------------------------------------------------------------------- #
# leakage guard — the most important test in the file
# --------------------------------------------------------------------------- #


def test_leakage_guard_raises_on_leaked_query() -> None:
    eval_queries = ["What is reciprocal rank fusion?", "How does DPO work?"]
    training_queries = ["how are chunks embedded", "What is reciprocal rank fusion?"]
    with pytest.raises(pairs_B.EvalLeakageError) as exc:
        pairs_B.assert_no_eval_leakage(training_queries, eval_queries)
    assert "reciprocal rank fusion" in str(exc.value).lower()


def test_leakage_guard_is_insensitive_to_case_and_whitespace() -> None:
    with pytest.raises(pairs_B.EvalLeakageError):
        pairs_B.assert_no_eval_leakage(["  WHAT   is  RAG? "], ["What is RAG?"])


def test_leakage_guard_passes_on_clean_data() -> None:
    pairs_B.assert_no_eval_leakage(
        ["how are chunks embedded", "what does bm25 weight"],
        ["What is reciprocal rank fusion?", "How does DPO work?"],
    )


def test_leakage_guard_runs_before_any_pair_is_written(tmp_path: Path) -> None:
    """A leak must abort the whole build: nothing on disk, non-zero exit upstream."""
    scores = tmp_path / "judge_scores.jsonl"
    queries = tmp_path / "queries_B.jsonl"
    out = tmp_path / "pairs.jsonl"
    leaked_query = "What is reciprocal rank fusion?"
    scores.write_text(
        "\n".join(
            make_score(leaked_query, f"c{i}", s).model_dump_json()
            for i, s in enumerate([3, 0], start=1)
        ),
        encoding="utf-8",
    )
    queries.write_text(json.dumps({"query": leaked_query}) + "\n", encoding="utf-8")

    cfg = {
        "pairs": {
            "judge_scores_path": str(scores),
            "eval_query_blocklist": str(queries),
            "out_path": str(out),
            "min_score_gap": 2,
            "max_pairs_per_query": 4,
        }
    }
    with pytest.raises(pairs_B.EvalLeakageError):
        pairs_B.build_pairs_from_config(cfg)
    assert not out.exists()


def test_missing_blocklist_is_an_error_not_an_empty_guard(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pairs_B.load_eval_queries(tmp_path / "nope.jsonl")


# --------------------------------------------------------------------------- #
# pair construction rules
# --------------------------------------------------------------------------- #


def test_min_score_gap_filters_judge_noise() -> None:
    scores = [make_score("q", "a", 3), make_score("q", "b", 2), make_score("q", "c", 0)]
    kept = pairs_B.build_pairs(scores, chunk_lookup=lookup, min_score_gap=2, max_pairs_per_query=10)
    assert {(p.chosen_chunk_id, p.rejected_chunk_id) for p in kept} == {("a", "c"), ("b", "c")}
    assert all(p.margin >= 2 for p in kept)

    gap_one = pairs_B.build_pairs(
        scores, chunk_lookup=lookup, min_score_gap=1, max_pairs_per_query=10
    )
    assert ("a", "b") in {(p.chosen_chunk_id, p.rejected_chunk_id) for p in gap_one}


def test_gap_of_three_leaves_only_the_widest_pairs() -> None:
    scores = [make_score("q", "a", 3), make_score("q", "b", 1), make_score("q", "c", 0)]
    kept = pairs_B.build_pairs(scores, chunk_lookup=lookup, min_score_gap=3, max_pairs_per_query=10)
    assert [(p.chosen_chunk_id, p.rejected_chunk_id) for p in kept] == [("a", "c")]


def test_per_query_cap_is_enforced_per_query() -> None:
    scores = [make_score("q1", f"a{i}", 3) for i in range(5)]
    scores += [make_score("q1", f"b{i}", 0) for i in range(5)]
    scores += [make_score("q2", f"c{i}", 3) for i in range(3)]
    scores += [make_score("q2", f"d{i}", 0) for i in range(3)]

    built = pairs_B.build_pairs(scores, chunk_lookup=lookup, min_score_gap=2, max_pairs_per_query=4)
    per_query: dict[str, int] = {}
    for pair in built:
        per_query[pair.query] = per_query.get(pair.query, 0) + 1
    assert per_query == {"q1": 4, "q2": 4}


def test_dedupe_drops_repeated_triples() -> None:
    rows = [make_score("q", "a", 3), make_score("q", "b", 0)]
    duplicated = rows + [make_score("Q  ", "a", 3), make_score("q", "b", 0)]

    deduped = pairs_B.build_pairs(
        duplicated, chunk_lookup=lookup, min_score_gap=2, max_pairs_per_query=10, dedupe=True
    )
    assert [(p.chosen_chunk_id, p.rejected_chunk_id) for p in deduped] == [("a", "b")]

    kept = pairs_B.build_pairs(
        duplicated, chunk_lookup=lookup, min_score_gap=2, max_pairs_per_query=10, dedupe=False
    )
    assert len(kept) > len(deduped)


def test_rejected_chunks_are_hard_negatives_from_the_same_candidate_set() -> None:
    scores = [
        make_score("query one", "q1_hi", 3),
        make_score("query one", "q1_lo", 0),
        make_score("query two", "q2_hi", 3),
        make_score("query two", "q2_lo", 0),
    ]
    built = pairs_B.build_pairs(
        scores, chunk_lookup=lookup, min_score_gap=2, max_pairs_per_query=10
    )
    candidates = {
        "query one": {"q1_hi", "q1_lo"},
        "query two": {"q2_hi", "q2_lo"},
    }
    assert built
    for pair in built:
        own = candidates[pair.query]
        assert pair.chosen_chunk_id in own
        assert pair.rejected_chunk_id in own, "rejected chunk escaped its own candidate set"


def test_a_chunk_is_never_paired_with_itself() -> None:
    scores = [make_score("q", "a", 3), make_score("q", "a", 0)]
    built = pairs_B.build_pairs(
        scores, chunk_lookup=lookup, min_score_gap=2, max_pairs_per_query=10
    )
    assert built == []


def test_pairs_round_trip_through_jsonl(tmp_path: Path) -> None:
    scores = [make_score("q", "a", 3), make_score("q", "b", 0)]
    built = pairs_B.build_pairs(scores, chunk_lookup=lookup, min_score_gap=2, max_pairs_per_query=4)
    out = pairs_B.write_pairs(built, tmp_path / "pairs.jsonl")
    assert train_B.load_pairs(out) == built


def test_build_pairs_rejects_a_nonsense_gap() -> None:
    with pytest.raises(ValueError):
        pairs_B.build_pairs([], min_score_gap=0)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_get_ranker_resolves_none_to_the_champion(tmp_registry: Path) -> None:
    ranker = registry_B.get_ranker(None, registry_file=tmp_registry)
    assert ranker.name == "baseline"
    assert isinstance(ranker, BaselineRanker)


def test_get_ranker_refuses_an_unknown_name(tmp_registry: Path) -> None:
    with pytest.raises(registry_B.UnknownRankerError):
        registry_B.get_ranker("no_such_model", registry_file=tmp_registry)
    # It is a KeyError subclass, so `except KeyError` upstream still works.
    with pytest.raises(KeyError):
        registry_B.get_ranker("no_such_model", registry_file=tmp_registry)


def test_register_never_overwrites_an_entry(tmp_registry: Path) -> None:
    registry_B.register(
        "dpo_v1", path="models/reranker_dpo_B", base_model="base", registry_file=tmp_registry
    )
    with pytest.raises(registry_B.DuplicateRankerError):
        registry_B.register(
            "dpo_v1", path="models/other", base_model="base", registry_file=tmp_registry
        )
    entry = registry_B.load_registry(tmp_registry)["entries"]["dpo_v1"]
    assert entry["path"] == "models/reranker_dpo_B", "the original entry survived"


def test_register_does_not_change_the_champion(tmp_registry: Path) -> None:
    registry_B.register("dpo_v1", path="models/x", registry_file=tmp_registry)
    assert registry_B.load_registry(tmp_registry)["champion"] == "baseline"


def test_promote_and_rollback_round_trip_through_the_json_file(tmp_registry: Path) -> None:
    registry_B.register(
        "dpo_v1",
        path="models/reranker_dpo_B",
        base_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        mlflow_run_id="abc123",
        metrics={"ndcg5": 0.41},
        registry_file=tmp_registry,
    )
    registry_B.promote("dpo_v1", registry_file=tmp_registry)

    on_disk = json.loads(tmp_registry.read_text(encoding="utf-8"))
    assert on_disk["champion"] == "dpo_v1"
    assert on_disk["entries"]["dpo_v1"]["metrics"] == {"ndcg5": 0.41}
    assert sorted(registry_B.list_rankers(tmp_registry)) == ["baseline", "dpo_v1"]

    # Rollback is one line: point the champion back at the kept baseline entry.
    registry_B.promote("baseline", registry_file=tmp_registry)
    rolled_back = json.loads(tmp_registry.read_text(encoding="utf-8"))
    assert rolled_back["champion"] == "baseline"
    assert "dpo_v1" in rolled_back["entries"], "rollback must not delete the challenger"


def test_promote_refuses_an_unknown_name(tmp_registry: Path) -> None:
    with pytest.raises(registry_B.UnknownRankerError):
        registry_B.promote("never_trained", registry_file=tmp_registry)


def test_cross_encoder_entry_raises_only_when_ranking_without_the_extra(
    tmp_registry: Path,
) -> None:
    registry_B.register("dpo_v1", path="models/absent", registry_file=tmp_registry)
    ranker = registry_B.get_ranker("dpo_v1", registry_file=tmp_registry)
    # Resolving is cheap and import-free; only `.rank` touches torch/the model dir.
    assert ranker.name == "dpo_v1"
    with pytest.raises((registry_B.RerankerDependencyError, FileNotFoundError)):
        ranker.rank("a query", [make_chunk("c1")])


def test_registry_module_imports_without_the_heavy_extra() -> None:
    assert train_B.missing_dependencies(), "this venv is expected to lack the train extra"
    assert registry_B.get_ranker(registry_file=None) is not None


# --------------------------------------------------------------------------- #
# the shared interface contract
# --------------------------------------------------------------------------- #


class StubTunedRanker:
    """Stands in for a trained cross-encoder: same interface, no torch."""

    name = "stub_tuned"

    def rank(self, query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
        scored = [(c, float(len(set(query.split()) & set(c.text.split())))) for c in chunks]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored


@pytest.mark.parametrize("ranker", [BaselineRanker(), StubTunedRanker()])
def test_rankers_satisfy_one_interface(ranker: Ranker) -> None:
    chunks = [make_chunk("c1", "alpha beta"), make_chunk("c2", "gamma"), make_chunk("c3", "beta")]
    ranked = ranker.rank("beta", chunks)

    assert isinstance(ranker, Ranker)
    assert isinstance(ranker.name, str) and ranker.name
    assert len(ranked) == len(chunks)
    assert {c.chunk_id for c, _ in ranked} == {c.chunk_id for c in chunks}
    assert all(isinstance(c, Chunk) and isinstance(s, float) for c, s in ranked)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranker.rank("beta", []) == []


def test_baseline_preserves_fusion_order() -> None:
    chunks = [make_chunk("c1"), make_chunk("c2"), make_chunk("c3")]
    ranked = BaselineRanker().rank("anything", chunks)
    assert [c.chunk_id for c, _ in ranked] == ["c1", "c2", "c3"]


def test_baseline_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError):
        BaselineRanker(mode="magic")


# --------------------------------------------------------------------------- #
# training path, without torch
# --------------------------------------------------------------------------- #


def test_train_settings_parse_the_real_config() -> None:
    settings = train_B.TrainSettings.from_config(registry_B.load_reranker_config())
    assert settings.base_model
    assert settings.beta > 0
    assert settings.output_dir.is_absolute()


def test_train_settings_reject_a_non_dpo_method() -> None:
    cfg = registry_B.load_reranker_config()
    cfg = {**cfg, "train": {**cfg["train"], "method": "hinge"}}
    with pytest.raises(ValueError):
        train_B.TrainSettings.from_config(cfg)


def test_dpo_rows_have_the_shape_trl_expects() -> None:
    scores = [make_score("q", "a", 3), make_score("q", "b", 0)]
    pairs = pairs_B.build_pairs(scores, chunk_lookup=lookup, min_score_gap=2)
    rows = train_B.build_dpo_rows(pairs)
    assert rows and all(set(r) == {"prompt", "chosen", "rejected"} for r in rows)


def test_empty_pairs_fail_loudly() -> None:
    with pytest.raises(ValueError):
        train_B.build_dpo_rows([])


def test_training_params_are_logged_with_the_provenance_needed_to_compare_runs() -> None:
    settings = train_B.TrainSettings.from_config(registry_B.load_reranker_config())
    params = train_B.training_params(settings, n_pairs=123)
    assert params["n_pairs"] == 123
    assert {"base_model", "beta", "lr", "epochs", "seed", "min_score_gap"} <= set(params)


def test_split_is_deterministic_for_a_seed() -> None:
    rows = [{"prompt": f"q{i}", "chosen": "a", "rejected": "b"} for i in range(20)]
    assert train_B.split_rows(rows, seed=7) == train_B.split_rows(rows, seed=7)
    train_rows, eval_rows = train_B.split_rows(rows, test_size=0.1, seed=7)
    assert len(train_rows) + len(eval_rows) == 20
    assert eval_rows


def test_invoking_training_without_the_extra_raises_an_actionable_error() -> None:
    with pytest.raises(train_B.TrainingDependenciesMissing) as exc:
        train_B.require_training_deps()
    message = str(exc.value)
    assert "[train]" in message and "pip install" in message
