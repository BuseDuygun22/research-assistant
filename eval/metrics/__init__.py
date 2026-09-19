"""Metric implementations.

- `retrieval_B` — Buse. nDCG@5, MRR, recall@20, citation precision. Promoted from
  notebook `06_metrics_baseline_B.ipynb`, which is the referee for Track A.
- `generation_S` — Sude. Faithfulness and answer relevance.

Submodules are intentionally not imported here: importing this package must stay
cheap and side-effect free, because Sude's gate runner imports one submodule only.
"""

from __future__ import annotations
