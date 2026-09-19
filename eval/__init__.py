"""Evaluation harnesses for the research assistant.

`eval.metrics.retrieval_B` (Buse, Track A) scores the retriever against the
hand-labelled held-out set. `eval.metrics.generation_S` (Sude, Track B) scores the
generated answer. `eval.run_gate_S` (Sude) reads `eval/thresholds_B.yaml` and imports
both to decide whether a change is allowed to ship.

This package lives at the repo root rather than under `src/` deliberately: it is
project infrastructure, not part of the shippable library, and nothing in
`src/research_assistant/` may import from it.
"""

from __future__ import annotations
