"""Configuration for the Clara MCP server (sevdesk backend)."""

from __future__ import annotations

import os


class Config:
    SERVER_NAME = "clara"
    HOST = os.getenv("MCP_HOST", "127.0.0.1")
    PORT = int(os.getenv("MCP_PORT", "8767"))

    SEVDESK_API_BASE = "https://my.sevdesk.de/api/v1"
    SEVDESK_API_TOKEN: str = os.environ["SEVDESK_API_TOKEN"]
