"""Langfuse tracing utilities.

Usage in route handlers:
    handler = build_langfuse_handler(session_id=session_id, trace_id=trace_id)
    config = make_runnable_config(handler)
    await graph.ainvoke(state, config=config)

When LANGFUSE_ENABLED=false (default), both functions are no-ops so call sites
need no conditional logic.
"""

from __future__ import annotations

from typing import Any

from librarian.config import settings
from core.logging import get_logger

log = get_logger(__name__)


def build_langfuse_handler(
    session_id: str,
    trace_id: str = "",
    user_id: str | None = None,
) -> Any | None:
    """Return a Langfuse CallbackHandler if tracing is enabled, else None.

    Parameters
    ----------
    session_id:
        Conversation identifier — groups related traces in Langfuse.
    trace_id:
        Request trace_id from RequestIDMiddleware, used as the Langfuse
        trace name so each HTTP request maps to one Langfuse trace.
    user_id:
        Optional — shown in the Langfuse user filter.
    """
    if not settings.langfuse_enabled:
        return None

    try:
        from langfuse.callback import CallbackHandler  # type: ignore[import-untyped]
    except ImportError:
        log.error("tracing.langfuse.missing", msg="langfuse not installed; tracing disabled")
        return None

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        log.error("tracing.langfuse.missing_keys")
        return None

    handler = CallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        session_id=session_id,
        user_id=user_id,
        trace_name=trace_id or session_id,
    )
    log.debug("tracing.langfuse.handler_created", session_id=session_id, trace_id=trace_id)
    return handler


def make_runnable_config(handler: Any | None) -> dict[str, Any]:
    """Build a LangGraph-compatible RunnableConfig dict with the Langfuse callback."""
    return {"callbacks": [handler] if handler else []}
