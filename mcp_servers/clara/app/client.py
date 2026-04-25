"""Shared async httpx client for the sevdesk API.

All tools import get_client() rather than creating their own client,
giving us a single connection pool with auth baked in.
"""

from __future__ import annotations

import httpx

from app.config import Config

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=Config.SEVDESK_API_BASE,
            headers={
                "Authorization": Config.SEVDESK_API_TOKEN,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=30.0,
        )
    return _client
