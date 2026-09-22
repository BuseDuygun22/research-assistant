"""Generate a synthetic paper corpus so every Track A stage is testable before real PDFs arrive.

What this fixture decides
-------------------------
`data/raw/` is empty and nothing may be downloaded, so the corpus that stages 01-06
are developed against is generated here. The generator is not a toy: it produces
conference-shaped PDFs with a large bold title, numbered bold section headings and
10pt body prose, because those are exactly the signals `parse_B.looks_like_heading`
keys on (see `notebooks/01_parsing_B.ipynb`). A fixture that did not reproduce the
font-size contrast would make the heading heuristic untestable.

Each paper is internally consistent and factually checkable: it names one method,
one dataset and one numeric result, and repeats none of them across papers. That is
what lets stage 05 build queries with known answers, and it is why the ground truth
written to `synthetic_ground_truth_B.json` records the *page* each fact landed on --
citation precision in stage 06 is measured against page ranges.

Ground-truth pages are not predicted from the layout, they are read back out of the
rendered PDF with pymupdf. Predicting them would silently rot the moment reportlab
reflowed a paragraph.

Run:
    .venv/Scripts/python.exe tests/fixtures/make_synthetic_corpus_B.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

REPO_ROOT = Path(__file__).resolve().parents[2]
# Kept out of data/raw/ deliberately: that directory holds the real corpus
# (see scripts/fetch_qasper_B.py, scripts/fetch_arxiv_fraud_B.py). These
# fixtures exist only for unit tests, which pass this path explicitly.
RAW_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic_pdfs_B"
MANIFEST = REPO_ROOT / "tests" / "fixtures" / "synthetic_manifest_B.jsonl"
GROUND_TRUTH = Path(__file__).resolve().parent / "synthetic_ground_truth_B.json"

BODY_SIZE = 10.0
HEADING_SIZE = 14.0
TITLE_SIZE = 17.0


@dataclass(frozen=True)
class PaperSpec:
    """One synthetic paper. Every field feeds both the rendered prose and the ground truth."""

    paper_id: str
    title: str
    year: int
    venue: str
    method: str
    method_family: str
    task: str
    dataset: str
    dataset_size: str
    metric: str
    value: str
    baseline: str
    baseline_value: str
    hardware: str
    limitation: str
    ablation: str
    ablation_delta: str
    paragraphs: dict[str, int] = field(default_factory=dict)


SPECS: list[PaperSpec] = [
    PaperSpec(
        paper_id="synth001",
        title="SparseLoom: Structured Sparsity for Long-Context Retrieval",
        year=2023,
        venue="Proceedings of the Workshop on Efficient Retrieval",
        method="SparseLoom",
        method_family="block-sparse attention",
        task="long-context passage retrieval",
        dataset="LoomBench",
        dataset_size="48,000",
        metric="nDCG@10",
        value="61.4",
        baseline="DenseWeave",
        baseline_value="54.9",
        hardware="a single 24GB consumer GPU",
        limitation="performance degrades on documents shorter than 200 tokens, where the "
        "block structure has too little material to exploit",
        ablation="removing the block router",
        ablation_delta="4.8",
        paragraphs={"intro": 5, "method": 5, "experiments": 5, "results": 4, "limitations": 2},
    ),
    PaperSpec(
        paper_id="synth002",
        title="QuillNet: Query Reformulation Without Supervised Pairs",
        year=2022,
        venue="Conference on Information Access",
        method="QuillNet",
        method_family="self-supervised reformulation",
        task="open-domain question answering",
        dataset="AskCorpus",
        dataset_size="112,300",
        metric="recall@20",
        value="78.2",
        baseline="TermExpand",
        baseline_value="71.0",
        hardware="four commodity CPU workers",
        limitation="the reformulator inherits the lexical biases of the corpus it was "
        "bootstrapped from, and transfers poorly to a new domain without a fresh bootstrap",
        ablation="disabling the round-trip consistency filter",
        ablation_delta="6.1",
        paragraphs={"intro": 4, "method": 4, "experiments": 4, "results": 3, "limitations": 2},
    ),
    PaperSpec(
        paper_id="synth003",
        title="Cascade Pruning for Neural Rerankers Under Latency Budgets",
        year=2024,
        venue="Symposium on Applied Ranking",
        method="CascadePrune",
        method_family="early-exit cascade",
        task="passage reranking",
        dataset="LatencyMSM",
        dataset_size="9,800",
        metric="MRR@10",
        value="39.7",
        baseline="FullCross",
        baseline_value="40.3",
        hardware="one server-class CPU socket",
        limitation="the cascade gives up 0.6 MRR@10 relative to the uncascaded cross-encoder, "
        "which is acceptable only where the latency budget is genuinely binding",
        ablation="fixing the exit threshold instead of learning it",
        ablation_delta="2.2",
        paragraphs={"intro": 7, "method": 7, "experiments": 7, "results": 5, "limitations": 3},
    ),
    PaperSpec(
        paper_id="synth004",
        title="GraniteIndex: Disk-Resident Vector Search for Small Teams",
        year=2023,
        venue="Workshop on Practical Vector Databases",
        method="GraniteIndex",
        method_family="disk-resident graph index",
        task="approximate nearest neighbour search",
        dataset="GraniteVec",
        dataset_size="2,400,000",
        metric="recall@10",
        value="94.6",
        baseline="FlatScan",
        baseline_value="99.9",
        hardware="a laptop with 16GB of RAM and an NVMe drive",
        limitation="recall is bought with disk bandwidth, so the index is unattractive on "
        "network-attached storage where random reads are expensive",
        ablation="removing the cached entry-point layer",
        ablation_delta="11.3",
        paragraphs={"intro": 5, "method": 6, "experiments": 5, "results": 4, "limitations": 3},
    ),
    PaperSpec(
        paper_id="synth005",
        title="Faithfulness Probes for Retrieval-Augmented Summarisation",
        year=2024,
        venue="Conference on Generation and Grounding",
        method="ProbeCheck",
        method_family="token-level attribution probe",
        task="grounded summarisation",
        dataset="GroundSum",
        dataset_size="15,600",
        metric="faithfulness F1",
        value="83.1",
        baseline="NLIJudge",
        baseline_value="76.4",
        hardware="a single mid-range GPU",
        limitation="the probe is trained on extractive summaries and its calibration drifts "
        "when the generator paraphrases heavily",
        ablation="collapsing the probe to a sentence-level classifier",
        ablation_delta="5.5",
        paragraphs={"intro": 4, "method": 5, "experiments": 4, "results": 4, "limitations": 2},
    ),
    PaperSpec(
        paper_id="synth006",
        title="Section-Aware Chunking Beats Fixed Windows on Scientific Text",
        year=2022,
        venue="Workshop on Document Understanding",
        method="SectionSplit",
        method_family="layout-conditioned segmentation",
        task="chunking for retrieval-augmented generation",
        dataset="SciChunk",
        dataset_size="3,100",
        metric="answer recall",
        value="72.9",
        baseline="FixedWindow512",
        baseline_value="65.2",
        hardware="a single CPU core",
        limitation="papers whose headings are not typographically distinct fall back to "
        "fixed windows, and on that subset the gain disappears entirely",
        ablation="dropping the title-and-section context prefix",
        ablation_delta="3.4",
        paragraphs={"intro": 6, "method": 6, "experiments": 6, "results": 5, "limitations": 2},
    ),
    PaperSpec(
        paper_id="synth007",
        title="RubricDPO: Preference Tuning a Reranker From Rubric Scores",
        year=2025,
        venue="Conference on Preference Learning",
        method="RubricDPO",
        method_family="direct preference optimisation",
        task="reranker training",
        dataset="RubricPairs",
        dataset_size="26,700",
        metric="nDCG@5",
        value="58.8",
        baseline="PointwiseBCE",
        baseline_value="52.1",
        hardware="two 40GB GPUs for ninety minutes",
        limitation="preference pairs inherit whatever the rubric judge is wrong about, so "
        "judge error becomes reranker error with no independent signal to catch it",
        ablation="sampling pairs uniformly instead of by score margin",
        ablation_delta="3.9",
        paragraphs={"intro": 7, "method": 7, "experiments": 7, "results": 5, "limitations": 3},
    ),
]


def _abstract(s: PaperSpec) -> list[str]:
    return [
        f"We present {s.method}, a {s.method_family} approach to {s.task}. "
        f"Existing systems treat this problem as a single monolithic decision, which "
        f"wastes computation on inputs that are easy and starves the inputs that are hard. "
        f"{s.method} instead allocates effort where the evidence is ambiguous. "
        f"On {s.dataset} we reach {s.value} {s.metric}, against {s.baseline_value} "
        f"{s.metric} for the {s.baseline} baseline, while remaining trainable on {s.hardware}.",
    ]


def _intro(s: PaperSpec, n: int) -> list[str]:
    pool = [
        f"Work on {s.task} has converged on a small number of architectures that differ "
        f"more in their training recipes than in their inductive biases. Progress reported "
        f"on public leaderboards is therefore difficult to attribute, and practitioners "
        f"choosing a system for a modest deployment have little guidance beyond parameter "
        f"count. We take the opposite starting point and ask what can be achieved when the "
        f"compute budget is fixed in advance at {s.hardware}.",
        f"Our contribution is {s.method}, which reorganises {s.task} around the observation "
        f"that most inputs are decided by a small fraction of the available evidence. "
        f"Rather than scaling the model, we make the decision procedure adaptive. This is a "
        f"deliberately unfashionable direction: it trades headline capacity for predictable "
        f"cost, and it is the trade practitioners actually face.",
        f"We evaluate on {s.dataset}, a collection of {s.dataset_size} examples assembled "
        f"for {s.task}. Against the {s.baseline} baseline, {s.method} improves {s.metric} "
        f"from {s.baseline_value} to {s.value}. An ablation shows that {s.ablation} costs "
        f"{s.ablation_delta} points of {s.metric}, which locates the gain in the component "
        f"we claim is responsible rather than in incidental tuning.",
        f"The remainder of this paper is organised as follows. Section 2 describes "
        f"{s.method} in detail. Section 3 sets out the experimental protocol and the "
        f"baselines we compare against. Section 4 reports results and ablations. Section 5 "
        f"discusses the limitations of the approach, of which the most serious is that "
        f"{s.limitation}.",
        f"Prior work on {s.task} divides roughly into two camps. The first scales the "
        f"encoder and accepts the resulting inference cost, on the assumption that capacity "
        f"is the binding constraint. The second compresses an expensive model into a cheap "
        f"one and accepts a fixed quality loss. Neither camp adapts the computation to the "
        f"individual input, which is the gap {s.method} occupies.",
        f"A second line of work, closer to ours, studies routing between models of different "
        f"sizes. Those systems typically route on features of the input alone. {s.method} "
        f"routes on the disagreement between two scorers that have both already looked at "
        f"the input, which is a strictly more informative signal and costs one cheap forward "
        f"pass to obtain.",
        f"We make three claims. First, that adaptive allocation recovers most of the quality "
        f"of an expensive model at a fraction of its cost on {s.dataset}. Second, that the "
        f"confidence estimator can be trained without human labels. Third, that the result "
        f"is reproducible on {s.hardware}, which we regard as a precondition for the claim "
        f"being useful rather than merely true.",
    ]
    return pool[:n]


def _method(s: PaperSpec, n: int) -> list[str]:
    pool = [
        f"{s.method} has three stages: a cheap scorer that produces a provisional decision, "
        f"a confidence estimator that decides whether the provisional decision is safe, and "
        f"an expensive scorer that is invoked only when it is not. The three stages share "
        f"an encoder, so the additional parameter cost over the {s.baseline} baseline is "
        f"under four percent.",
        f"The confidence estimator is the component that makes the {s.method_family} "
        f"framing pay. We train it with a margin objective against the disagreement between "
        f"the cheap and expensive scorers, which requires no additional human annotation: "
        f"the supervision is manufactured from the two scorers that already exist.",
        "Training proceeds in two phases. The cheap and expensive scorers are trained "
        "jointly for the first phase, then frozen while the estimator is fitted. Freezing "
        "matters. When all three are trained end to end the estimator learns to mark "
        "everything as uncertain, because that minimises loss without ever being penalised "
        "for the compute it spends.",
        f"Inference is a single forward pass for the majority of inputs. On {s.dataset} the "
        f"expensive scorer fires on roughly one input in five, which is what makes {s.method} "
        f"runnable on {s.hardware} at all. We provide the exact thresholds and the "
        f"schedule used for every reported number in the supplementary material.",
        "The cheap scorer is a two-layer projection over pooled encoder states. We chose "
        "pooling over a learned attention head after finding that the two performed within "
        "noise of one another on validation, and pooling has no parameters to tune. Where "
        "two designs tie, we take the one with fewer knobs, because every knob is a future "
        "source of irreproducibility.",
        f"The expensive scorer is a full cross-attention pass over the input pair. It is the "
        f"same architecture as {s.baseline}, deliberately, so that any difference in the "
        f"reported {s.metric} is attributable to the routing and not to a better backbone. "
        f"We initialise both scorers from the same checkpoint for the same reason.",
        "One design detail is worth stating because it cost us several weeks. The "
        "confidence estimator must be given the cheap scorer's logits, not its probabilities. "
        "With probabilities the estimator saturates on confident inputs and loses the "
        "ordering information it needs near the decision boundary, which is precisely the "
        "region where routing decisions matter.",
    ]
    return pool[:n]


def _experiments(s: PaperSpec, n: int) -> list[str]:
    pool = [
        f"We evaluate on {s.dataset}, which contains {s.dataset_size} examples. We use the "
        f"standard split released with the dataset and report {s.metric} on the held-out "
        f"test portion. No test example is used for threshold selection; thresholds are "
        f"chosen on a validation split carved out of the training portion.",
        f"Our primary baseline is {s.baseline}, which we retrain under our own protocol "
        f"rather than quoting published numbers. Retraining moves the {s.baseline} figure by "
        f"more than a point relative to the number reported by its authors, which is itself "
        f"a small argument for retraining baselines rather than copying tables.",
        f"All runs use {s.hardware}. Every reported figure is the mean of five seeds. We "
        f"report the mean rather than the best seed because the seed-to-seed spread on "
        f"{s.dataset} is roughly one point of {s.metric}, which is large enough that a "
        f"best-of-five protocol would manufacture most of a claimed improvement.",
        f"We additionally run an ablation in which {s.ablation} is applied, holding every "
        f"other setting fixed. This isolates the contribution of the component we claim is "
        f"doing the work, and it is reported alongside the main result in the next section.",
        f"Hyperparameters were selected by random search over sixteen configurations on the "
        f"validation split, with the search budget fixed in advance and identical for "
        f"{s.method} and for {s.baseline}. Equalising the search budget matters more than "
        f"the search method: an unequal budget is the most common way a comparison on "
        f"{s.dataset} becomes unfair without anyone intending it.",
        f"We also measure wall-clock cost. Every latency figure is taken on {s.hardware} "
        f"with a warm cache and reported as the median over one thousand queries, since the "
        f"tail of the distribution is dominated by allocator behaviour rather than by the "
        f"model. Tail latency is reported separately in the supplementary material.",
        f"Statistical testing uses a paired bootstrap over the test queries of {s.dataset} "
        f"with ten thousand resamples. We report a difference as meaningful only when the "
        f"bootstrap interval excludes zero, which on this dataset requires a gap of roughly "
        f"one point of {s.metric}.",
    ]
    return pool[:n]


def _results(s: PaperSpec, n: int) -> list[str]:
    pool = [
        f"{s.method} reaches {s.value} {s.metric} on {s.dataset}. The {s.baseline} baseline "
        f"reaches {s.baseline_value} {s.metric} under the identical protocol. The gap is "
        f"stable across all five seeds and does not depend on the choice of validation split.",
        f"The ablation confirms where the gain comes from. With {s.ablation}, {s.metric} "
        f"falls by {s.ablation_delta} points, recovering most of the distance back to "
        f"{s.baseline}. We read this as evidence that the improvement is a property of the "
        f"{s.method_family} design rather than of the training schedule.",
        f"Cost is the other half of the claim. {s.method} completes the full {s.dataset} "
        f"test pass on {s.hardware}, which the uncascaded configuration does not. For a "
        f"team without cluster access this is the difference between a system that can be "
        f"evaluated and one that can only be cited.",
        f"Breaking the {s.dataset} test set down by input length shows the gain is not "
        f"uniform. The longest quartile of inputs accounts for most of the improvement, "
        f"which is consistent with the {s.method_family} account: long inputs are where the "
        f"cheap scorer is least reliable and where routing therefore has the most to decide.",
        f"We ran the paired bootstrap described in Section 3 over the difference between "
        f"{s.method} and {s.baseline}. The interval excludes zero comfortably, so we regard "
        f"the headline gap as real rather than as seed noise. The ablation gap of "
        f"{s.ablation_delta} points clears the same bar.",
    ]
    return pool[:n]


def _limitations(s: PaperSpec, n: int) -> list[str]:
    pool = [
        f"The most serious limitation of {s.method} is that {s.limitation}. We report this "
        f"prominently because it determines whether the method is appropriate for a given "
        f"deployment, and it is not visible from the headline {s.metric} figure.",
        f"A second limitation is evaluation scope. {s.dataset} is a single dataset in a "
        f"single language, and we make no claim that the {s.ablation_delta} point ablation "
        f"gap would reproduce elsewhere. Cross-dataset transfer is the obvious next study "
        f"and we have not run it.",
        f"Finally, the thresholds that govern when the expensive scorer fires were tuned on "
        f"{s.dataset} and are likely to need retuning on any new corpus. We have not "
        f"investigated whether they can be set adaptively at inference time.",
    ]
    return pool[:n]


def _references(s: PaperSpec) -> list[str]:
    return [
        f"[1] A. Renner and B. Okafor. Scaling laws for {s.task}. Journal of Retrieval "
        f"Systems, 2021.",
        f"[2] C. Lindqvist. {s.baseline}: a strong baseline for {s.task}. In Proceedings of "
        f"the Conference on Information Access, 2020.",
        f"[3] D. Marchetti and E. Yilmaz. The {s.dataset} collection. Technical report, 2021.",
        "[4] F. Novak. On the reproducibility of leaderboard results. Transactions on "
        "Empirical Methods, 2022.",
        "[5] G. Adeyemi and H. Sorensen. Evaluation protocols that survive contact with "
        "practice. Annual Review of Applied Machine Learning, 2023.",
    ]


def build_sections(s: PaperSpec) -> list[tuple[str, list[str]]]:
    """Section headings exactly as they must render, paired with their body paragraphs."""
    p = s.paragraphs
    return [
        ("Abstract", _abstract(s)),
        ("1 Introduction", _intro(s, p.get("intro", 3))),
        ("2 Method", _method(s, p.get("method", 3))),
        ("3 Experiments", _experiments(s, p.get("experiments", 3))),
        ("4 Results", _results(s, p.get("results", 3))),
        ("5 Limitations", _limitations(s, p.get("limitations", 2))),
        ("References", _references(s)),
    ]


def render_pdf(spec: PaperSpec, out_path: Path) -> None:
    """Render one paper. Font sizes are the whole point: 10pt body, 14pt bold headings."""
    title_style = ParagraphStyle(
        "SynthTitle", fontName="Helvetica-Bold", fontSize=TITLE_SIZE, leading=TITLE_SIZE + 4,
        spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "SynthHeading", fontName="Helvetica-Bold", fontSize=HEADING_SIZE,
        leading=HEADING_SIZE + 3, spaceBefore=12, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "SynthBody", fontName="Helvetica", fontSize=BODY_SIZE, leading=BODY_SIZE + 3.5,
        alignment=TA_JUSTIFY, spaceAfter=6,
    )

    doc = SimpleDocTemplate(
        str(out_path), pagesize=LETTER,
        leftMargin=1.0 * inch, rightMargin=1.0 * inch,
        topMargin=1.0 * inch, bottomMargin=1.0 * inch,
        title=spec.title, author="Synthetic Corpus Generator",
    )
    flow: list[Any] = [Paragraph(spec.title, title_style)]
    flow.append(Paragraph(f"{spec.venue}, {spec.year}", body_style))
    flow.append(Spacer(1, 10))
    for heading, paras in build_sections(spec):
        flow.append(Paragraph(heading, heading_style))
        for para in paras:
            flow.append(Paragraph(para, body_style))
    doc.build(flow)


def _normalise(text: str) -> str:
    return " ".join(text.split())


def locate_facts(pdf_path: Path, spec: PaperSpec) -> list[dict[str, Any]]:
    """Read the rendered PDF back and record which page each checkable fact landed on.

    Reading back rather than predicting is deliberate: reportlab decides pagination, and
    a predicted page number would go stale on any prose edit without failing loudly.
    """
    doc = pymupdf.open(pdf_path)
    pages = [_normalise(page.get_text()) for page in doc]
    doc.close()

    facts = [
        {
            "fact_id": f"{spec.paper_id}-method",
            "kind": "method_name",
            "question": f"What method does '{spec.title}' introduce?",
            "answer": spec.method,
            "evidence": (
                f"We present {spec.method}, a {spec.method_family} "
                f"approach to {spec.task}."
            ),
        },
        {
            "fact_id": f"{spec.paper_id}-dataset",
            "kind": "dataset_name",
            "question": f"Which dataset is {spec.method} evaluated on?",
            "answer": spec.dataset,
            "evidence": (
                f"We evaluate on {spec.dataset}, which contains "
                f"{spec.dataset_size} examples."
            ),
        },
        {
            "fact_id": f"{spec.paper_id}-result",
            "kind": "numeric_result",
            "question": f"What {spec.metric} does {spec.method} achieve on {spec.dataset}?",
            "answer": f"{spec.value} {spec.metric}",
            "evidence": f"{spec.method} reaches {spec.value} {spec.metric} on {spec.dataset}.",
        },
        {
            "fact_id": f"{spec.paper_id}-baseline",
            "kind": "baseline_comparison",
            "question": f"What does the {spec.baseline} baseline achieve on {spec.dataset}?",
            "answer": f"{spec.baseline_value} {spec.metric}",
            "evidence": f"The {spec.baseline} baseline reaches {spec.baseline_value} "
            f"{spec.metric} under the identical protocol.",
        },
        {
            "fact_id": f"{spec.paper_id}-ablation",
            "kind": "ablation",
            "question": f"How much {spec.metric} is lost when {spec.ablation} in {spec.method}?",
            "answer": f"{spec.ablation_delta} points of {spec.metric}",
            "evidence": (
                f"With {spec.ablation}, {spec.metric} falls by "
                f"{spec.ablation_delta} points"
            ),
        },
    ]

    for fact in facts:
        needle = _normalise(str(fact["evidence"]))
        hits = [i + 1 for i, page in enumerate(pages) if needle in page]
        if not hits:
            raise RuntimeError(
                f"{pdf_path.name}: evidence for {fact['fact_id']} not found in rendered text. "
                "A paragraph probably broke across a page and the sentence was hyphenated."
            )
        fact["page"] = hits[0]
        fact["pages"] = hits
    return facts


def generate(out_dir: Path = RAW_DIR) -> dict[str, Any]:
    """Render every spec, write the corpus manifest, and return the ground-truth payload."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    papers: list[dict[str, Any]] = []

    for spec in SPECS:
        filename = f"{spec.paper_id}_{spec.method.lower()}.pdf"
        path = out_dir / filename
        render_pdf(spec, path)

        doc = pymupdf.open(path)
        n_pages = doc.page_count
        doc.close()

        manifest_rows.append(
            {
                "filename": filename,
                "paper_id": spec.paper_id,
                "title": spec.title,
                "year": spec.year,
                "venue": spec.venue,
                "why_included": "Synthetic fixture: exercises section-aware chunking and "
                "gives stage 05 a query with a known answer.",
            }
        )
        papers.append(
            {
                "paper_id": spec.paper_id,
                "filename": filename,
                "title": spec.title,
                "year": spec.year,
                "venue": spec.venue,
                "n_pages": n_pages,
                "method": spec.method,
                "dataset": spec.dataset,
                "metric": spec.metric,
                "value": spec.value,
                "sections": [h for h, _ in build_sections(spec)],
                "facts": locate_facts(path, spec),
            }
        )

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as fh:
        for row in manifest_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload = {
        "generator": "tests/fixtures/make_synthetic_corpus_B.py",
        "note": "Synthetic corpus. Page numbers are 1-based and read back from the rendered "
        "PDF, so they are valid for exactly these files.",
        "body_font_size": BODY_SIZE,
        "heading_font_size": HEADING_SIZE,
        "dropped_section_heading": "References",
        "n_papers": len(papers),
        "papers": papers,
    }
    GROUND_TRUTH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    payload = generate()
    print(f"wrote {payload['n_papers']} synthetic papers to {RAW_DIR}")
    for paper in payload["papers"]:
        print(
            f"  {paper['filename']:34s} pages={paper['n_pages']}  "
            f"method={paper['method']:14s} dataset={paper['dataset']}"
        )
    total = sum(p["n_pages"] for p in payload["papers"])
    print(f"total pages: {total}")
    print(f"manifest:     {MANIFEST}")
    print(f"ground truth: {GROUND_TRUTH}")


if __name__ == "__main__":
    main()
