"""Generation-side eval metrics (Sude).

Three families, and the split matters because they fail independently:

* **Attribution** — is the draft supported by its sources? (`citation_precision`,
  `citation_coverage`)
* **Utility** — does the draft answer the question, and did it abstain when it
  should have? (`answer_relevance`, `abstention_*`)
* **Process** — did the agent get there sensibly? (`trajectory_metrics`)

The reason the utility family exists at all: gating on attribution alone rewards
the degenerate strategy. Five verbatim quotes stitched together score 1.0
citation precision and 1.0 faithfulness while answering nothing, so a change that
made drafts more timid would read as an improvement. Attribution is a safety
property; it is not a quality property, and a gate that confuses the two
optimises for silence.

What is *not* here, and cannot be without new labels: **answer correctness**. A
draft can be perfectly attributed, perfectly relevant, and false — because its
source is wrong, or because retrieval surfaced one side of a disagreement.
Measuring that needs reference answers, which `EvalQuery` does not carry. The
limit is named in `docs/design_review_J.md` rather than hidden behind metrics
that sound like they cover it.

Stdlib only, same as `significance_S.py`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

# --- attribution -------------------------------------------------------------


def citation_precision(verified: int, cited: int) -> float:
    """Cited claims that are actually supported / cited claims.

    Undefined when nothing is cited. Returns 0.0 rather than 1.0 for that case:
    an uncited draft is the worst outcome for this metric, and the vacuous-truth
    reading (`0/0 = 1.0`) would score it perfect. Getting this backwards is the
    single easiest way to build a gate that rewards citing nothing.
    """
    if cited <= 0:
        return 0.0
    return verified / cited


def citation_coverage(cited: int, total_claims: int) -> float:
    """Claims carrying any citation / total claims.

    Coverage of the *draft*. Not coverage of the *corpus* — see `context_recall`,
    which is the other direction and the one that catches cherry-picking.
    """
    if total_claims <= 0:
        return 0.0
    return cited / total_claims


# --- utility -----------------------------------------------------------------


def answer_relevance(grades: Sequence[int]) -> float:
    """Mean answer-relevance grade, normalised to [0, 1].

    Grades are 0-3 on the same TREC-style scale as retrieval relevance, so the
    two are readable side by side in a results table.
    """
    if not grades:
        return 0.0
    return fmean(grades) / 3.0


@dataclass(frozen=True)
class AbstentionScores:
    """Abstention measured as a classification problem, which is what it is.

    A single "abstention rate" is uninterpretable — it conflates the system that
    correctly declines an unanswerable question with the one that declines
    everything. These four cells separate them.
    """

    correct_abstentions: int  # unanswerable, abstained     — right
    missed_abstentions: int  # unanswerable, answered      — the dangerous cell
    over_abstentions: int  # answerable, abstained       — timid
    correct_answers: int  # answerable, answered        — right

    @property
    def total(self) -> int:
        return (
            self.correct_abstentions
            + self.missed_abstentions
            + self.over_abstentions
            + self.correct_answers
        )

    @property
    def abstention_recall(self) -> float:
        """Of the questions that should have been declined, how many were?

        The headline number. A miss here is a confidently wrong answer to a
        question the corpus cannot support, which is the failure mode with the
        worst consequences in a research assistant.
        """
        should = self.correct_abstentions + self.missed_abstentions
        return self.correct_abstentions / should if should else 0.0

    @property
    def abstention_precision(self) -> float:
        """Of the abstentions made, how many were warranted?

        Guards the opposite failure: a system that abstains constantly scores
        perfect recall and is useless.
        """
        made = self.correct_abstentions + self.over_abstentions
        return self.correct_abstentions / made if made else 0.0

    @property
    def over_abstention_rate(self) -> float:
        answerable = self.over_abstentions + self.correct_answers
        return self.over_abstentions / answerable if answerable else 0.0


def score_abstention(
    outcomes: Sequence[tuple[bool, bool]],
) -> AbstentionScores:
    """Score (was_unanswerable, did_abstain) pairs.

    `was_unanswerable` comes from `EvalQuery.intent == "unanswerable"`, which is
    why that label has to exist in the eval set for this to be measurable at all.
    """
    c = Counter(outcomes)
    return AbstentionScores(
        correct_abstentions=c[(True, True)],
        missed_abstentions=c[(True, False)],
        over_abstentions=c[(False, True)],
        correct_answers=c[(False, False)],
    )


def context_recall(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Of the chunks the qrels mark relevant, how many did retrieval surface?

    This is the cheapest guard against the faithful-but-misleading draft. A draft
    can cite every one of its claims correctly and still mislead, if retrieval
    only surfaced one side of a disagreement — each claim supported, the picture
    wrong. Coverage of the draft cannot see that; coverage of the corpus can.

    Costs no new labels: it reads the qrels Buse is already producing. Note this
    is a *retrieval* property being consumed on the generation side — the
    canonical implementation belongs in `eval/metrics/retrieval_B.py` when Buse
    writes it, and this should then defer to hers rather than the two drifting.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0  # nothing to recall; not a failure
    return len(relevant & set(retrieved_ids)) / len(relevant)


# --- granularity ---------------------------------------------------------------


@dataclass(frozen=True)
class GranularityProfile:
    """How finely a draft was decomposed into claims.

    Measured rather than assumed. The rubric asks for atomic claims, and the
    citation-granularity work reports that attribution quality *peaks at an
    intermediate granularity* — splitting too finely fractures the dependencies a
    claim needs, so each fragment is individually supported while the join between
    them goes unchecked. That is also the mechanism behind "A. B. Therefore C."
    passing claim-by-claim. Whether our judge sits on the good or bad side of that
    peak is an empirical question, and this is the instrument for asking it.
    """

    claims_per_sentence: float
    mean_words_per_claim: float
    n_claims: int
    n_sentences: int


_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z0-9]+")


def claim_granularity(draft: str, claims: Sequence[str]) -> GranularityProfile:
    """Profile one draft's decomposition."""
    sentences = [s for s in _SENTENCE.split(draft.strip()) if s.strip()]
    n_sent = max(1, len(sentences))
    words = [len(_WORD.findall(c)) for c in claims]
    return GranularityProfile(
        claims_per_sentence=len(claims) / n_sent,
        mean_words_per_claim=fmean(words) if words else 0.0,
        n_claims=len(claims),
        n_sentences=len(sentences),
    )


def granularity_sweep(
    precision_by_level: dict[str, Sequence[float]],
) -> tuple[str, dict[str, float]]:
    """Which decomposition level gives the best citation precision?

    Run the judge at several granularities over the same drafts — e.g. "atomic",
    "clause", "sentence" — and pass the per-draft precision for each. Returns the
    best level and every level's mean. Deliberately a comparison over the *same*
    drafts: comparing granularities across different drafts would confound the
    decomposition with the content.
    """
    if not precision_by_level:
        raise ValueError("no granularity levels to compare")
    lengths = {len(v) for v in precision_by_level.values()}
    if len(lengths) != 1:
        raise ValueError("every level must be scored on the same drafts")
    means = {level: fmean(v) if v else 0.0 for level, v in precision_by_level.items()}
    best = max(means, key=lambda k: means[k])
    return best, means


# --- process -----------------------------------------------------------------


def trajectory_metrics(trajectories: Sequence[Sequence[dict[str, Any]]]) -> dict[str, float]:
    """Measure how the agent got there, not only where it arrived.

    A system can reach an acceptable answer by a broken path — three retrievals
    that each changed nothing, an escalation that fired on a bug rather than on
    genuine uncertainty — and a gate that only reads the final draft will promote
    it. These are the numbers that make the revision cap an empirical question
    instead of a guess.
    """
    if not trajectories:
        return {}
    triggers: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    rewrites: list[int] = []
    re_retrievals: list[int] = []
    lengths: list[int] = []

    for t in trajectories:
        lengths.append(len(t))
        rewrites.append(sum(1 for s in t if s["route"] == "rewrite"))
        re_retrievals.append(sum(1 for s in t if s["route"] == "re_retrieve"))
        for s in t:
            triggers[s["trigger"]] += 1
            routes[s["route"]] += 1

    n = len(trajectories)
    out: dict[str, float] = {
        "mean_steps": fmean(lengths),
        "mean_rewrites": fmean(rewrites),
        "mean_re_retrievals": fmean(re_retrievals),
        # Accepted with no revision of either kind. Not `len(t) == 1`: triage adds a
        # step to every run, so step count no longer measures revisions.
        "first_pass_accept_rate": sum(
            1
            for t in trajectories
            if t and t[-1]["route"] == "accept"
            and not any(s["route"] in ("rewrite", "re_retrieve") for s in t)
        )
        / n,
        "escalation_rate": routes["escalate"] / n,
        "abstention_rate": routes["abstain"] / n,
    }
    for trigger, count in triggers.items():
        out[f"trigger.{trigger}"] = count / n
    return out


def retrieval_gain(before: float, after: float) -> float:
    """Change in retrieval quality across one reformulation round.

    Persistently near zero means the researcher is spending budget without
    improving evidence, which is an argument for cutting the re-retrieval cap
    rather than raising it — and is invisible to any endpoint metric.
    """
    return after - before
