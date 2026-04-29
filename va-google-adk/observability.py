"""Observability config for va-google-adk.

LangSmith (active)
------------------
ADK doesn't use LangChain so auto-instrumentation doesn't apply here.
We use the LangSmith Client directly to create one trace per ADK turn.

Env vars required (same as va-langgraph):
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_pt_...
    LANGSMITH_PROJECT=billy-va
    LANGSMITH_ENDPOINT=https://api.smith.langchain.com

Langfuse (reference — swap back by replacing deps/env vars and uncommenting below)
----------------------------------------------------------------------------------
# from langfuse import Langfuse
#
# _langfuse_client = None
#
# def init_langfuse() -> None:
#     global _langfuse_client
#     if os.getenv("LANGFUSE_ENABLED", "").lower() != "true":
#         return
#     public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
#     secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
#     if not public_key or not secret_key:
#         return
#     _langfuse_client = Langfuse(
#         public_key=public_key,
#         secret_key=secret_key,
#         host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
#     )
#
# def shutdown_langfuse() -> None:
#     if _langfuse_client is not None:
#         _langfuse_client.flush()
#
# def start_trace(trace_id, user_id, session_id, input):
#     if _langfuse_client is None:
#         return None
#     return _langfuse_client.trace(
#         id=trace_id, user_id=user_id, session_id=session_id, input=input, name="adk-turn",
#     )
#
# In gateway/main.py lifespan: init_langfuse() / shutdown_langfuse()
# In session_manager.run_turn: lf_trace = start_trace(...); lf_trace.update(output=...) on success
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import structlog

log = structlog.get_logger(__name__)

_langsmith_client = None


def init_langsmith() -> None:
    global _langsmith_client
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
        log.info("langsmith-disabled", reason="LANGSMITH_TRACING != true")
        return
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    if not api_key:
        log.warning("langsmith-missing-api-key")
        return
    try:
        from langsmith import Client

        _langsmith_client = Client(
            api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
            api_key=api_key,
        )
        log.info("langsmith-ready", project=os.getenv("LANGSMITH_PROJECT", "billy-va"))
    except Exception:
        log.exception("langsmith-init-failed")


def shutdown_langsmith() -> None:
    # LangSmith client flushes automatically; nothing to do explicitly
    pass


class _Turn:
    """Thin wrapper around a LangSmith run for one ADK turn."""

    def __init__(self, run_id: uuid.UUID) -> None:
        self._run_id = run_id

    def finish(self, output: str | None, error: str | None = None) -> None:
        if _langsmith_client is None:
            return
        try:
            _langsmith_client.update_run(
                run_id=self._run_id,
                outputs={"response": output} if output else None,
                error=error,
                end_time=datetime.now(timezone.utc),
            )
        except Exception:
            log.exception("langsmith-run-update-failed")


def start_trace(
    trace_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    input: str | None = None,
) -> _Turn | None:
    """Open a LangSmith run for one ADK turn. Returns a _Turn to close, or None if disabled."""
    if _langsmith_client is None:
        return None
    try:
        run_id = uuid.UUID(trace_id) if trace_id else uuid.uuid4()
        _langsmith_client.create_run(
            id=run_id,
            name="adk-turn",
            run_type="chain",
            inputs={"message": input},
            project_name=os.getenv("LANGSMITH_PROJECT", "billy-va"),
            tags=["adk"],
            extra={"session_id": session_id, "user_id": user_id},
            start_time=datetime.now(timezone.utc),
        )
        return _Turn(run_id)
    except Exception:
        log.exception("langsmith-start-trace-failed")
        return None
