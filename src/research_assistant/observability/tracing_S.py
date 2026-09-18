"""Tracing (Sude — see docs/contracts_J.md open question 2).

Design constraint: instrumentation must never be the reason something breaks.
Langfuse is an optional import and an optional runtime dependency; with tracing
off, `@traced` costs one attribute lookup and `span()` is a no-op object. That
is what makes it safe to decorate every tool, judge call and agent node rather
than sprinkling instrumentation only where a bug has already been found.

Buse instruments ingestion/retrieval with the same primitives so a single trace
covers the whole request, not just the agent graph.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from research_assistant.config_J import get_settings

logger = logging.getLogger("research_assistant.tracing")

F = TypeVar("F", bound=Callable[..., Any])

# Span kinds. Keeping them a closed set means traces can be grouped in the UI
# without relying on everyone spelling "retrieval" the same way.
KIND_INGESTION = "ingestion"
KIND_RETRIEVAL = "retrieval"
KIND_TOOL = "tool"
KIND_JUDGE = "judge"
KIND_AGENT = "agent"
KIND_LLM = "llm"


class _NoopSpan:
    """Duck-typed stand-in used whenever tracing is off or Langfuse is absent."""

    def __init__(self, name: str, kind: str) -> None:
        self.name = name
        self.kind = kind
        self.trace_id = "noop"

    def set(self, **_: Any) -> None:  # pragma: no cover - trivial
        pass

    def end(self, **_: Any) -> None:  # pragma: no cover - trivial
        pass


class _LocalSpan(_NoopSpan):
    """Structured-log span. The fallback when tracing is enabled but Langfuse is
    not configured — a local dev run still gets timings and inputs in the log,
    which is most of the debugging value at zero setup cost."""

    def __init__(self, name: str, kind: str, trace_id: str) -> None:
        super().__init__(name, kind)
        self.trace_id = trace_id
        self._t0 = time.perf_counter()
        self._fields: dict[str, Any] = {}

    def set(self, **fields: Any) -> None:
        self._fields.update(fields)

    def end(self, **fields: Any) -> None:
        self._fields.update(fields)
        elapsed = (time.perf_counter() - self._t0) * 1000
        logger.info(
            "span kind=%s name=%s trace=%s ms=%.1f %s",
            self.kind,
            self.name,
            self.trace_id,
            elapsed,
            self._fields,
        )


_langfuse_client: Any = None
_langfuse_tried = False


def _client() -> Any:
    """Lazily construct the Langfuse client once. A failure here downgrades to
    local spans and is logged — it never propagates to the caller."""
    global _langfuse_client, _langfuse_tried
    if _langfuse_tried:
        return _langfuse_client
    _langfuse_tried = True
    s = get_settings()
    if not (s.tracing_enabled and s.langfuse_public_key and s.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
        )
    except Exception as exc:  # noqa: BLE001 - instrumentation must not break callers
        logger.warning("Langfuse unavailable, falling back to local spans: %s", exc)
        _langfuse_client = None
    return _langfuse_client


@contextmanager
def span(name: str, kind: str = KIND_TOOL, **inputs: Any) -> Iterator[Any]:
    """Open a span. Always yields something with `.set()` and `.end()`."""
    s = get_settings()
    if not s.tracing_enabled:
        yield _NoopSpan(name, kind)
        return

    trace_id = uuid.uuid4().hex[:12]
    client = _client()
    handle: Any = _LocalSpan(name, kind, trace_id)
    lf_span = None
    if client is not None:
        try:
            lf_span = client.trace(name=name, metadata={"kind": kind}, input=inputs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse span failed, using local span: %s", exc)

    handle.set(**inputs)
    try:
        yield handle
    except Exception as exc:
        handle.end(status="error", error=repr(exc))
        if lf_span is not None:
            try:
                lf_span.update(level="ERROR", status_message=repr(exc))
            except Exception:  # noqa: BLE001
                pass
        raise
    else:
        handle.end(status="ok")
        if lf_span is not None:
            try:
                lf_span.update(output=getattr(handle, "_fields", {}).get("output"))
            except Exception:  # noqa: BLE001
                pass


def traced(name: str | None = None, kind: str = KIND_TOOL) -> Callable[[F], F]:
    """Decorator form of `span`, for sync and async callables alike.

    Arguments are deliberately NOT auto-captured: chunk text is large and can be
    sensitive. Callers record what matters with `span.set(...)`.
    """

    def decorate(fn: F) -> F:
        span_name = name or fn.__qualname__

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                with span(span_name, kind) as sp:
                    result = await fn(*args, **kwargs)
                    sp.set(output=_summarise(result))
                    return result

            return awrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(span_name, kind) as sp:
                result = fn(*args, **kwargs)
                sp.set(output=_summarise(result))
                return result

        return wrapper  # type: ignore[return-value]

    return decorate


def _summarise(value: Any) -> Any:
    """Keep spans small. A trace full of chunk bodies is unreadable and expensive."""
    if isinstance(value, (str, bytes)):
        return f"<{type(value).__name__} len={len(value)}>"
    if isinstance(value, (list, tuple, set)):
        return f"<{type(value).__name__} n={len(value)}>"
    if hasattr(value, "model_dump"):
        return type(value).__name__
    return value


def flush() -> None:
    """Called at process exit / end of a CI job so buffered events are not lost."""
    client = _client()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse flush failed: %s", exc)
