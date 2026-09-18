"""Tool deadlines (Sude).

The one failure class the error design did not cover. `ToolError` handles a tool
that *fails*; nothing handled a tool that simply does not return. An agent
blocked on a hung backend consumes no budget, produces no violation, and trips no
escalation — it just stops, which is the worst of the available failure modes
because nothing in the system can see it.

A deadline converts that into an ordinary, routable `ToolError`: retryable, with
a remediation that tells the agent to ask for less rather than to ask again
identically.

Implementation note: this runs the call on a worker thread and abandons it on
timeout. Python cannot safely kill a thread, so an abandoned call keeps running
to completion in the background — it is *orphaned*, not cancelled. That is
acceptable here because the operations behind it are reads, so the cost of an
orphan is wasted work rather than a corrupted write. It would not be acceptable
for anything that mutates state, and this module should not be used for that.
The alternative — an async client with real cancellation — is the right fix if
the tool layer ever becomes async.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# One shared pool. Threads are cheap relative to the retrieval calls they wrap,
# and a per-call pool would leak a thread on every timeout.
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool-deadline")


class ToolTimeout(TimeoutError):
    """A tool exceeded its deadline. Carries the budget it blew, so the caller
    can put a useful number in the remediation rather than saying 'too slow'."""

    def __init__(self, operation: str, seconds: float) -> None:
        super().__init__(f"{operation} exceeded its {seconds:.1f}s deadline")
        self.operation = operation
        self.seconds = seconds


def with_deadline(operation: str, seconds: float, fn: Callable[[], T]) -> T:
    """Run `fn`, raising `ToolTimeout` if it outlasts `seconds`.

    A deadline of zero or less disables the check, which is what the tests use to
    keep the thread pool out of the way.
    """
    if seconds <= 0:
        return fn()
    future: Future[T] = _POOL.submit(fn)
    try:
        return future.result(timeout=seconds)
    except FutureTimeout:
        future.cancel()  # only works if it never started; otherwise it orphans
        logger.warning("%s exceeded its %.1fs deadline; abandoning the call", operation, seconds)
        raise ToolTimeout(operation, seconds) from None
