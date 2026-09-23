"""One question in, one cited answer out (Sude): the whole pipeline behind a CLI.

    python -m research_assistant.ask_S "Which methods handle class imbalance?"
    python -m research_assistant.ask_S --llm ollama --model qwen2.5:7b-instruct "..."
    python -m research_assistant.ask_S --mlflow --json "..."

The run is: retrieve (hybrid BM25 + dense, RRF, rerank) -> triage the evidence ->
write a cited draft -> judge it (faithfulness, answer relevance) -> route (accept /
rewrite / re-retrieve / abstain / escalate). This module only wires the flags,
resolves the `[chunk_id]` citations in the accepted draft back to papers and pages,
and reports. It adds no decisions of its own.

Exit codes are meaningful so a script or CI step can act on them:
0 answered (accepted), 3 abstained (the corpus does not contain an answer),
4 escalated (a person is needed). Anything else is a crash.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Sequence
from typing import Any

EXIT_CODES = {"accepted": 0, "abstained": 3, "escalated": 4}
_CITATION = re.compile(r"\[([0-9a-fA-F]{6,})\]")


def _apply_overrides(args: argparse.Namespace) -> None:
    """Flags win over `.env`. Must run before `get_settings()` is first called."""
    if args.llm:
        os.environ["RA_JUDGE_BACKEND"] = args.llm
    if args.model:
        os.environ["RA_JUDGE_MODEL"] = args.model
    if args.retrieval:
        os.environ["RA_RETRIEVAL_BACKEND"] = args.retrieval
    if args.llm == "ollama" and "RA_TOOL_DEADLINE_SECONDS" not in os.environ:
        # The 10 s default is sized for hosted APIs; a local 7B model on CPU or a
        # small GPU needs longer for the same call, and a timeout here is reported
        # as a retrieval-style failure that hides the real cause.
        os.environ["RA_TOOL_DEADLINE_SECONDS"] = "0"


def resolve_citations(draft: str) -> list[dict[str, Any]]:
    """Look every `[chunk_id]` in the draft up in the index, in order of appearance.

    Unresolvable ids are returned with `found: False` rather than dropped: a
    citation to nothing is exactly what a reader must be told about.
    """
    from research_assistant.mcp_server.backend_S import get_backend

    backend = get_backend()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cid in _CITATION.findall(draft):
        if cid in seen:
            continue
        seen.add(cid)
        chunk = backend.get_chunk(cid)
        if chunk is None:
            out.append({"chunk_id": cid, "found": False})
            continue
        m = chunk.metadata
        out.append(
            {
                "chunk_id": cid,
                "found": True,
                "paper_id": m.paper_id,
                "title": m.title,
                "year": m.year,
                "page": m.page,
                "section": m.section,
                "quote": chunk.text.strip()[:300],
            }
        )
    return out


def ask(question: str) -> dict[str, Any]:
    """Run the agent graph and return a plain, JSON-serialisable report."""
    from research_assistant.agents.graph_S import run
    from research_assistant.config_J import get_settings
    from research_assistant.llm_S import get_llm
    from research_assistant.mcp_server.backend_S import get_backend

    settings = get_settings()
    backend = get_backend()
    t0 = time.perf_counter()
    state, handover = run(question)
    seconds = time.perf_counter() - t0

    draft = state.draft.text if state.draft else None
    report: dict[str, Any] = {
        "question": question,
        "outcome": state.outcome,
        "answer": draft if state.outcome == "accepted" else None,
        "references": resolve_citations(draft) if draft and state.outcome == "accepted" else [],
        "steps": len(state.decisions),
        "rewrites_used": state.budget.rewrites_used,
        "re_retrievals_used": state.budget.re_retrievals_used,
        "queries": list(state.queries_issued),
        "n_evidence": len(state.evidence),
        "faithfulness_passed": state.faithfulness.passed if state.faithfulness else None,
        "answer_relevance": state.answer.relevance if state.answer else None,
        "handover": None
        if handover is None
        else {
            "trigger": handover.trigger,
            "reason": handover.reason,
            "next_step": handover.next_step,
            "draft": handover.draft,
        },
        "llm_backend": settings.judge_backend,
        "llm_model": getattr(get_llm(), "model", settings.judge_model),
        "retrieval_backend": type(backend).__name__,
        "corpus_version": settings.corpus_version,
        "seconds": round(seconds, 2),
        "run_id": state.run_id,
    }
    return report


def log_to_mlflow(report: dict[str, Any], tracking_uri: str, experiment: str) -> str:
    """Record one question as an MLflow run: config as params, outcome as metrics,
    the full report as an artifact. Lets an agent run sit next to the retrieval eval
    and training runs it depends on."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"ask-{report['run_id']}") as run:
        mlflow.log_params(
            {
                "llm_backend": report["llm_backend"],
                "llm_model": report["llm_model"],
                "retrieval_backend": report["retrieval_backend"],
                "corpus_version": report["corpus_version"],
            }
        )
        mlflow.set_tag("outcome", report["outcome"])
        metrics = {
            "seconds": report["seconds"],
            "steps": report["steps"],
            "n_evidence": report["n_evidence"],
            "n_references": len(report["references"]),
            "accepted": 1.0 if report["outcome"] == "accepted" else 0.0,
        }
        if report["answer_relevance"] is not None:
            metrics["answer_relevance"] = float(report["answer_relevance"])
        mlflow.log_metrics(metrics)
        mlflow.log_text(json.dumps(report, indent=2, ensure_ascii=False), "report.json")
        return str(run.info.run_id)


def render(report: dict[str, Any]) -> str:
    lines = [f"Q: {report['question']}", ""]
    if report["outcome"] == "accepted":
        lines += [report["answer"] or "", "", "References:"]
        for r in report["references"]:
            if r["found"]:
                page = f", p. {r['page']}" if r["page"] else ""
                lines.append(f"  [{r['chunk_id']}] {r['title']} ({r['year'] or 'n.d.'}){page}")
            else:
                lines.append(f"  [{r['chunk_id']}] NOT FOUND IN INDEX")
    else:
        h = report["handover"] or {}
        label = "ABSTAINED" if report["outcome"] == "abstained" else "ESCALATED TO A PERSON"
        lines += [
            label,
            f"  why:  {h.get('reason', 'n/a')}",
            f"  next: {h.get('next_step', 'n/a')}",
        ]
        if h.get("draft"):
            lines += ["", "Unverified draft (do not rely on it):", h["draft"]]
    lines += [
        "",
        f"[{report['outcome']}] {report['steps']} routing steps, {report['n_evidence']} passages, "
        f"{report['seconds']}s | llm={report['llm_backend']}:{report['llm_model']} "
        f"retrieval={report['retrieval_backend']} corpus={report['corpus_version']}",
    ]
    return "\n".join(lines)


def _ensure_utf8_stdout() -> None:
    """A retrieved passage or a draft can contain any Unicode character a paper
    used (math notation, Greek letters, em dashes) - `print` must not depend on
    the console's legacy codepage to render it. Windows terminals and redirected
    output default to cp1252, which cannot encode most of that and raises
    `UnicodeEncodeError` deep inside `print`, crashing an otherwise-successful
    run. `reconfigure` is a no-op where stdout is already UTF-8 (most of Linux/
    macOS), and it is unavailable on a stream some odd runner has replaced -
    fall through silently rather than let a cosmetic guard break the real work.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Ask the research assistant one question.")
    ap.add_argument("question", help="the research question")
    ap.add_argument("--llm", choices=["stub", "ollama", "anthropic", "gemini"], default=None)
    ap.add_argument(
        "--model", default=None, help="model id / Ollama tag (e.g. qwen2.5:7b-instruct)"
    )
    ap.add_argument("--retrieval", choices=["auto", "stub", "track_a"], default=None)
    ap.add_argument("--json", dest="as_json", action="store_true", help="print the report as JSON")
    ap.add_argument("--mlflow", action="store_true", help="log this run to MLflow")
    ap.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    ap.add_argument("--mlflow-experiment", default="research-assistant-agent")
    args = ap.parse_args(argv)

    _apply_overrides(args)
    report = ask(args.question)
    if args.mlflow:
        report["mlflow_run_id"] = log_to_mlflow(report, args.mlflow_uri, args.mlflow_experiment)
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.as_json else render(report))
    return EXIT_CODES.get(report["outcome"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
