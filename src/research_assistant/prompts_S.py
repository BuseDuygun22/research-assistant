"""Worked response examples appended to every structured-output prompt (Sude).

Why this exists: a hosted frontier model follows "return a JSON object matching the
schema" from the schema alone. A local 3B-14B model (Qwen, via Ollama) often does
not - it wraps the JSON in prose, invents field names, or answers in the wrong
shape. Showing one or two complete input/output pairs per task is the cheapest fix
and it costs nothing on a model that did not need it.

Rules the examples follow, because a local model copies whatever it is shown:

- **Every example is valid against the real schema.** `tests/unit/test_prompt_examples_S.py`
  parses each one with the same pydantic model the pipeline uses, so an example can
  never drift from the contract.
- **Examples cover the hard case, not only the easy one.** Each task shows a clean
  pass *and* a failure or abstention, otherwise the model learns that the answer is
  always "everything is fine".
- **Placeholder ids are obviously fake** (`a1b2c3`), and the prompt says so, so an
  example id is never mistaken for a real chunk.

`with_examples(system, task)` is the single entry point. Unknown tasks raise, so a
typo cannot silently ship an example-free prompt.
"""

from __future__ import annotations

import json
from typing import Any

_FOOTER = (
    "\n\n## Response format\n"
    "Reply with ONE JSON object and nothing else: no prose before or after, no "
    "markdown code fence, no comments. Use exactly the field names shown in the "
    "examples. The ids and text in the examples are illustrations only - never copy "
    "them into your answer; use the ids that appear in the passages you are given."
)


def _dump(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


# task -> list of (situation, input excerpt, output object)
EXAMPLES: dict[str, list[tuple[str, str, dict[str, Any]]]] = {
    "write": [
        (
            "Passages support the answer",
            "QUESTION: Which model family did the paper use for card-fraud detection?\n"
            "PASSAGES:\n[a1b2c3] (Graph fraud study, Methods)\n"
            "We build a graph neural network over the transaction graph and train it "
            "on the labelled edges.\n",
            {
                "draft": "The paper builds a graph neural network over the transaction "
                "graph [a1b2c3].",
                "claims": [
                    {
                        "text": "The paper builds a graph neural network over the "
                        "transaction graph [a1b2c3].",
                        "chunk_id": "a1b2c3",
                    }
                ],
            },
        ),
        (
            "Passages do not cover the question - say so, do not invent",
            "QUESTION: What was the F1 score on the Kaggle dataset?\n"
            "PASSAGES:\n[d4e5f6] (Survey, Introduction)\n"
            "Fraud detection is a central problem for payment providers.\n",
            {
                "draft": "The retrieved passages do not report an F1 score on that dataset.",
                "claims": [],
            },
        ),
    ],
    "revise": [
        (
            "One claim was flagged overstated; the other is kept untouched",
            "CURRENT DRAFT: Method X reduces fraud losses [a1b2c3]. It uses a "
            "transformer [d4e5f6].\n"
            "CLAIMS TO FIX:\n- [overstated_certainty] Method X reduces fraud losses "
            "[a1b2c3].\n  diagnosis: source says 'may reduce'\n"
            "KEEP EXACTLY: ['It uses a transformer [d4e5f6].']",
            {
                "draft": "Method X may reduce fraud losses [a1b2c3]. It uses a "
                "transformer [d4e5f6].",
                "claims": [
                    {
                        "text": "Method X may reduce fraud losses [a1b2c3].",
                        "chunk_id": "a1b2c3",
                    },
                    {"text": "It uses a transformer [d4e5f6].", "chunk_id": "d4e5f6"},
                ],
            },
        ),
    ],
    "reformulate": [
        (
            "First retry: change the vocabulary, not the wording",
            "QUESTION: How do banks catch stolen-card transactions?\n"
            "ALREADY TRIED:\n- how do banks catch stolen-card transactions",
            {
                "query": "credit card fraud detection supervised classifier imbalanced data",
                "rationale": "Papers say 'credit card fraud detection', not 'stolen card'.",
            },
        ),
    ],
    "triage": [
        (
            "Passages answer the question",
            "QUESTION: What does the paper use to handle class imbalance?\n"
            "PASSAGES:\n[a1b2c3] (Paper A, 2022)\nWe apply SMOTE oversampling to the "
            "minority class before training.",
            {
                "label": "sufficient",
                "conflicts": [],
                "missing_aspects": [],
                "confidence": 0.9,
                "rationale": "The passage names the technique directly.",
            },
        ),
        (
            "On topic but does not answer - insufficient, not partial",
            "QUESTION: What latency does the system achieve?\n"
            "PASSAGES:\n[d4e5f6] (Paper B, 2021)\nThe system is deployed at a large "
            "payment processor.",
            {
                "label": "insufficient",
                "conflicts": [],
                "missing_aspects": ["reported inference latency"],
                "confidence": 0.85,
                "rationale": "Deployment is mentioned but no latency figure is given.",
            },
        ),
        (
            "Two passages disagree",
            "QUESTION: Does oversampling improve recall?\n"
            "PASSAGES:\n[a1b2c3] (Paper A, 2020)\nSMOTE improved recall by 8 points.\n"
            "[d4e5f6] (Paper B, 2023)\nSMOTE did not change recall on our data.",
            {
                "label": "conflicting",
                "conflicts": [
                    {
                        "kind": "disagreement",
                        "chunk_ids": ["a1b2c3", "d4e5f6"],
                        "summary": "One reports a recall gain from SMOTE, the other none.",
                        "newer_chunk_id": None,
                    }
                ],
                "missing_aspects": [],
                "confidence": 0.75,
                "rationale": "Both are current studies and they disagree.",
            },
        ),
    ],
    "faithfulness": [
        (
            "Clean draft: no violations",
            "RETRIEVED CONTEXT:\n[a1b2c3] (Paper A)\nThe model reaches 0.91 AUC.\n"
            "DRAFT: The model reaches 0.91 AUC [a1b2c3].",
            {
                "violations": [],
                "total_claims": 1,
                "cited_claims": 1,
                "supported_cited_claims": 1,
                "confidence": 0.9,
            },
        ),
        (
            "Overstated claim: report the sentence verbatim",
            "RETRIEVED CONTEXT:\n[a1b2c3] (Paper A)\nResults suggest the approach may "
            "lower false positives.\n"
            "DRAFT: The approach lowers false positives [a1b2c3].",
            {
                "violations": [
                    {
                        "kind": "overstated_certainty",
                        "claim": "The approach lowers false positives [a1b2c3].",
                        "cited_chunk_ids": ["a1b2c3"],
                        "explanation": "Source says 'suggest' and 'may'; draft asserts flatly.",
                        "severity": "major",
                    }
                ],
                "total_claims": 1,
                "cited_claims": 1,
                "supported_cited_claims": 0,
                "confidence": 0.8,
            },
        ),
    ],
    "answer": [
        (
            "Fluent draft over irrelevant evidence - abstention signal",
            "QUESTION: What was the false-positive rate?\n"
            "RETRIEVED CONTEXT:\n[a1b2c3] Fraud losses reached billions last year.\n"
            "DRAFT: Fraud is costly [a1b2c3].",
            {
                "relevance": 0,
                "evidence_sufficient": False,
                "missing_aspects": ["reported false-positive rate of the method"],
                "confidence": 0.85,
                "rationale": "The draft is on topic but never answers the question.",
            },
        ),
        (
            "Fully answered",
            "QUESTION: Which dataset was used?\n"
            "RETRIEVED CONTEXT:\n[a1b2c3] Experiments use the IEEE-CIS dataset.\n"
            "DRAFT: The experiments use the IEEE-CIS dataset [a1b2c3].",
            {
                "relevance": 3,
                "evidence_sufficient": True,
                "missing_aspects": [],
                "confidence": 0.9,
                "rationale": "The draft names the dataset, as the passage does.",
            },
        ),
    ],
    "relevance": [
        (
            "Directly answers",
            "QUESTION: Which loss function is used for training?\n"
            "PASSAGE (a1b2c3):\nWe train with focal loss to down-weight easy negatives.",
            {"grade": 3, "confidence": 0.9, "rationale": "Names the loss function."},
        ),
        (
            "Same topic, no answer",
            "QUESTION: Which loss function is used for training?\n"
            "PASSAGE (d4e5f6):\nFraud detection systems must handle imbalanced data.",
            {
                "grade": 1,
                "confidence": 0.8,
                "rationale": "Mentions imbalance, which is background, not the loss used.",
            },
        ),
    ],
}


def render_examples(task: str) -> str:
    """The examples block for one task, as prompt text."""
    try:
        examples = EXAMPLES[task]
    except KeyError as exc:
        raise KeyError(f"no examples for task {task!r}; known: {sorted(EXAMPLES)}") from exc
    parts = ["\n\n## Examples"]
    for i, (situation, given, out) in enumerate(examples, 1):
        parts.append(f"\n### Example {i} - {situation}\nInput:\n{given}\n\nOutput:\n{_dump(out)}")
    return "\n".join(parts)


def with_examples(system: str, task: str) -> str:
    """`system` plus worked examples and the response-format footer."""
    return system.rstrip() + render_examples(task) + _FOOTER
