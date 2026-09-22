"""The few-shot examples must stay valid against the schemas the pipeline parses with.

A stale example is worse than none: a local model copies it faithfully, and the
copy then fails validation in production. Parsing each one with the real reply
model turns a contract change into a failing test instead of a flaky judge.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from research_assistant.agents.nodes.researcher_S import _Reformulation
from research_assistant.agents.nodes.writer_S import _DraftOut
from research_assistant.judge.faithfulness_S import _AnswerReply, _FaithfulnessReply
from research_assistant.judge.relevance_S import _RelevanceReply
from research_assistant.judge.triage_S import _TriageReply
from research_assistant.prompts_S import EXAMPLES, render_examples, with_examples

SCHEMAS: dict[str, type[BaseModel]] = {
    "write": _DraftOut,
    "revise": _DraftOut,
    "reformulate": _Reformulation,
    "triage": _TriageReply,
    "faithfulness": _FaithfulnessReply,
    "answer": _AnswerReply,
    "relevance": _RelevanceReply,
}


def test_every_task_has_a_schema_and_vice_versa() -> None:
    assert set(EXAMPLES) == set(SCHEMAS)


@pytest.mark.parametrize(
    ("task", "index"),
    [(task, i) for task, items in EXAMPLES.items() for i in range(len(items))],
)
def test_example_parses_against_the_real_schema(task: str, index: int) -> None:
    _situation, _given, output = EXAMPLES[task][index]
    SCHEMAS[task].model_validate(output)


def test_write_examples_keep_claim_text_verbatim_in_draft() -> None:
    """The editor matches claims to the draft by string equality (rule 4)."""
    for task in ("write", "revise"):
        for _s, _g, out in EXAMPLES[task]:
            for claim in out["claims"]:
                assert claim["text"] in out["draft"]


def test_every_task_shows_a_non_trivial_case() -> None:
    """A task with only the happy path teaches the model that nothing ever fails."""
    for task in ("triage", "faithfulness", "answer", "relevance", "write"):
        assert len(EXAMPLES[task]) >= 2, task


def test_with_examples_appends_format_footer() -> None:
    out = with_examples("SYSTEM", "relevance")
    assert out.startswith("SYSTEM")
    assert "## Examples" in out
    assert "ONE JSON object" in out


def test_unknown_task_raises() -> None:
    with pytest.raises(KeyError):
        render_examples("nonexistent")
