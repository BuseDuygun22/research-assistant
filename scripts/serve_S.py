"""Entrypoints for the two servers (Sude).

`serve-api` runs the FastAPI tool service; `serve-mcp` runs the MCP adapter over
it. Two processes rather than one because the split is the architecture: the
logic is HTTP, the MCP layer only translates protocol.

Both print the readiness state on startup. A service that comes up on a stub
backend and says nothing is how unpublishable numbers end up in a report.
"""

from __future__ import annotations

import argparse
import logging
import sys

from research_assistant.config_J import get_settings
from research_assistant.mcp_server.health_S import readiness

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a research-assistant server.")
    parser.add_argument("target", choices=["api", "mcp"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    status = readiness()
    banner = f"backend={status.backend} status={status.status} corpus={status.corpus_version}"
    if status.status != "ok":
        logger.warning("DEGRADED: %s - %s", banner, status.detail)
    else:
        logger.info(banner)

    if args.target == "api":
        import uvicorn

        uvicorn.run(
            "research_assistant.mcp_server.api_S:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    else:
        from research_assistant.mcp_server.server_S import main as mcp_main

        logger.info("MCP transport=%s", get_settings().mcp_transport)
        mcp_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
