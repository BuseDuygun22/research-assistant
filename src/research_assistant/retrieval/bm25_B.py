"""Sparse arm: tokenisation, BM25 index, persistence.

Decides: what a token is (lowercase alphanumeric runs, English stopwords removed),
that the index is an in-memory `rank_bm25.BM25Okapi` rebuilt from a pickled
tokenised corpus, and that the index doubles as the local chunk catalogue.

The sparse arm exists because dense retrieval silently misses exact identifiers --
a model name, a dataset name, a metric abbreviation. The reasoning, and the ablation
that decides whether this arm earns its keep, are in
`notebooks/04_hybrid_retrieval_B.ipynb` (design-choice table: sparse arm).

Config: `configs/retrieval_B.yaml` -> `bm25:`.

Two deviations from the notebook prototype:

1. The notebook pickles the live `BM25Okapi` object. That pickle is tied to the
   installed `rank_bm25` version and to its internal attribute names, so a dependency
   bump silently produces an unloadable index. This module pickles the tokenised
   corpus, the ids, the chunks and the `k1`/`b` parameters, and reconstructs the
   scorer on load. Rebuilding is milliseconds at this corpus size.
2. The notebook keeps only `chunk_id`s. The index here retains the `Chunk`s, which is
   what lets `service_B.get_section` enumerate a paper's section without a second
   store and without widening the `VectorStore` interface.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..config_J import load_config, resolve_path
from ..contracts.retrieval_J import Chunk

# Deliberately small and closed-class: function words that carry no topical signal in
# an academic corpus. Kept short on purpose -- an aggressive list starts eating terms
# like "no", "not" and "than" that do discriminate in a methods section.
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the of and or to in for on with is are was were be been being this that these
    those we our us it its as by from at can could may might must shall should will
    would not no nor than then so such but if into onto over under about above below
    there here when where which who whom whose what why how all any both each few more
    most other some only own same too very do does did done have has had having
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_PICKLE_FORMAT = 2


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Lowercase, split on non-alphanumerics, optionally drop English stopwords.

    Digits are kept inside tokens so that `bert-base`, `f1`, `gpt4` and `bleu4`
    survive as single searchable units -- exactly the terms the dense arm loses.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if not remove_stopwords:
        return tokens
    return [t for t in tokens if t not in ENGLISH_STOPWORDS]


class BM25Index:
    """BM25Okapi over chunk texts, plus the chunks themselves.

    Construct with `BM25Index.build(chunks)`, persist with `save()`, reload with
    `BM25Index.load()`. `search` returns chunk ids; `search_scored` also returns the
    raw BM25 scores, which the weighted-sum fusion needs and RRF does not.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        tokenized: Sequence[Sequence[str]],
        *,
        k1: float,
        b: float,
        remove_stopwords: bool = True,
    ) -> None:
        if len(chunks) != len(tokenized):
            raise ValueError(f"chunks/tokens length mismatch: {len(chunks)} != {len(tokenized)}")
        self.chunks: list[Chunk] = list(chunks)
        self.ids: list[str] = [c.chunk_id for c in self.chunks]
        self.by_id: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}
        self.tokenized: list[list[str]] = [list(t) for t in tokenized]
        self.k1 = float(k1)
        self.b = float(b)
        self.remove_stopwords = bool(remove_stopwords)
        self._scorer: Any | None = None

    # -- construction -----------------------------------------------------

    @classmethod
    def build(
        cls,
        chunks: Iterable[Chunk],
        *,
        k1: float | None = None,
        b: float | None = None,
        stopwords: str | None = None,
    ) -> BM25Index:
        """Tokenise and index `chunks`. Unspecified parameters come from config."""
        if k1 is None or b is None or stopwords is None:
            cfg = load_config("retrieval")["bm25"]
            k1 = k1 if k1 is not None else cfg["k1"]
            b = b if b is not None else cfg["b"]
            stopwords = stopwords if stopwords is not None else cfg.get("stopwords")
        remove = str(stopwords).lower() == "english"
        chunk_list = list(chunks)
        tokenized = [tokenize(c.text, remove_stopwords=remove) for c in chunk_list]
        return cls(
            chunk_list,
            tokenized,
            k1=float(k1),
            b=float(b),
            remove_stopwords=remove,
        )

    def _get_scorer(self) -> Any:
        if self._scorer is None:
            from rank_bm25 import BM25Okapi

            # BM25Okapi divides by average document length, so an all-empty corpus
            # (or an empty one) must not reach it.
            corpus = [toks or ["\x00empty"] for toks in self.tokenized]
            if not corpus:
                corpus = [["\x00empty"]]
            self._scorer = BM25Okapi(corpus, k1=self.k1, b=self.b)
        return self._scorer

    # -- query ------------------------------------------------------------

    def search_scored(self, query: str, n: int) -> list[tuple[str, float]]:
        """Top `n` `(chunk_id, bm25_score)`, best first. Zero-scoring hits are dropped.

        Dropping zeros matters: `rank_bm25` scores every document, so without this a
        query with two rare terms would still hand fusion 30 documents, 28 of which
        contain none of the query terms and would dilute the candidate set.
        """
        if n <= 0 or not self.ids:
            return []
        tokens = tokenize(query, remove_stopwords=self.remove_stopwords)
        if not tokens:
            return []
        scores = self._get_scorer().get_scores(tokens)
        ranked = sorted(
            ((cid, float(s)) for cid, s in zip(self.ids, scores, strict=True) if s > 0.0),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:n]

    def search(self, query: str, n: int) -> list[str]:
        """Top `n` chunk ids for `query`, best first."""
        return [cid for cid, _ in self.search_scored(query, n)]

    def get(self, chunk_id: str) -> Chunk:
        try:
            return self.by_id[chunk_id]
        except KeyError:
            raise KeyError(chunk_id) from None

    def __len__(self) -> int:
        return len(self.ids)

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Pickle the index. Defaults to `bm25.index_path` from config."""
        target = resolve_path(path if path is not None else _default_index_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": _PICKLE_FORMAT,
            "chunks": [c.model_dump() for c in self.chunks],
            "tokenized": self.tokenized,
            "k1": self.k1,
            "b": self.b,
            "remove_stopwords": self.remove_stopwords,
        }
        with target.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return target

    @classmethod
    def load(cls, path: str | Path | None = None) -> BM25Index:
        """Load a pickled index and rebuild the scorer."""
        source = resolve_path(path if path is not None else _default_index_path())
        if not source.exists():
            raise FileNotFoundError(f"BM25 index not found at {source}; build it first")
        with source.open("rb") as fh:
            payload = pickle.load(fh)
        fmt = payload.get("format")
        if fmt != _PICKLE_FORMAT:
            raise ValueError(
                f"BM25 index at {source} has format {fmt!r}, "
                f"expected {_PICKLE_FORMAT!r}; rebuild it"
            )
        return cls(
            [Chunk(**row) for row in payload["chunks"]],
            payload["tokenized"],
            k1=payload["k1"],
            b=payload["b"],
            remove_stopwords=payload["remove_stopwords"],
        )


def _default_index_path() -> str:
    return str(load_config("retrieval")["bm25"]["index_path"])
