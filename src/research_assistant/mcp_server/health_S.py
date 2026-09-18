"""Health and readiness (Sude).

Liveness and readiness are kept apart on purpose. The container is *live* as soon
as the process answers; it is *ready* only once a retrieval backend can actually
serve a query. Collapsing the two would let the CI deploy step go green while the
index is missing, which is exactly the failure the eval gate exists to catch.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from research_assistant.config_J import get_settings
from research_assistant.contracts import CONTRACT_VERSION
from research_assistant.contracts.retrieval_J import RetrievalRequest


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded", "error"]
    contract_version: str
    corpus_version: str
    backend: str
    detail: str | None = None


def liveness() -> HealthStatus:
    return HealthStatus(
        status="ok",
        contract_version=CONTRACT_VERSION,
        corpus_version=get_settings().corpus_version,
        backend="n/a",
    )


def readiness() -> HealthStatus:
    """Issue one real retrieval. A backend that imports but cannot answer is not
    ready, and only a live query distinguishes the two.

    A stub backend reports `degraded`, never `ok`: the service works, but any
    numbers it produces are not publishable, and the deploy job keys off this.
    """
    from research_assistant.mcp_server.backend_S import StubRetrieval, get_backend

    try:
        backend = get_backend()
        probe = backend.retrieve(RetrievalRequest(query="health probe", top_k=1))
    except Exception as exc:  # noqa: BLE001 - health must never raise
        return HealthStatus(
            status="error",
            contract_version=CONTRACT_VERSION,
            corpus_version=get_settings().corpus_version,
            backend="unknown",
            detail=repr(exc),
        )

    is_stub = isinstance(backend, StubRetrieval)
    return HealthStatus(
        status="degraded" if is_stub else "ok",
        contract_version=CONTRACT_VERSION,
        corpus_version=probe.corpus_version,
        backend="stub" if is_stub else "track-a",
        detail="stub backend - results are not publishable" if is_stub else None,
    )
