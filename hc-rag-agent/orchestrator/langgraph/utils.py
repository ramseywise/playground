"""Shared helpers: async bridge for LangGraph nodes, end-to-end latency total."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")

_LATENCY_KEYS = (
    "query_transform_ms",
    "retrieval_ms",
    "rerank_ms",
    "policy_retrieval_ms",
    "policy_rerank_ms",
    "llm_ms",
    "post_answer_eval_ms",
)


def run_coro(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` whether or not an event loop is already running (e.g. FastAPI)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

    return asyncio.run(coro)


def with_total_ms(latency_ms: dict[str, float]) -> dict[str, float]:
    """Attach ``total_ms`` — sum of known pipeline stages (milliseconds)."""
    out = dict(latency_ms)
    out["total_ms"] = sum(out.get(k, 0.0) for k in _LATENCY_KEYS)
    return out


__all__ = ["run_coro", "with_total_ms"]
