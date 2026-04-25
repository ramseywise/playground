"""Clara MCP server — sevdesk backend.

Run in STDIO mode (default, for Claude Desktop / ADK MCPToolset):
    python -m app.main_noauth

Run in SSE/HTTP mode (for remote agents or browser testing):
    python -m app.main_noauth --http
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from fastmcp import FastMCP

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from app.common import register_all
from app.config import Config

mcp = FastMCP(name=Config.SERVER_NAME)
register_all(mcp)


def run_server() -> None:
    if "--http" in sys.argv:
        sys.stderr.write(
            f"Clara MCP server (SSE) listening on http://{Config.HOST}:{Config.PORT}/sse\n"
        )
        mcp.run(transport="sse", host=Config.HOST, port=Config.PORT)
    else:
        mcp.run()


if __name__ == "__main__":
    run_server()
