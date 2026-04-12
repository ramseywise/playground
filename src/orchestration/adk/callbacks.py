"""ADK agent and tool callbacks for observability.

Provides structured logging AND optional Langfuse tracing for every
agent invocation and tool call, giving ADK agents the same observability
as the LangGraph pipeline.

When ``LANGFUSE_ENABLED=true``, traces are emitted to Langfuse with
agent/tool spans. When disabled, only structlog events are emitted.
"""

from __future__ import annotations

import time
from typing import Any

from google.adk.agents.context import Context
from google.adk.tools.base_tool import BaseTool
from google.genai import types

from core.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Langfuse client (lazy, optional)
# ---------------------------------------------------------------------------

_langfuse: Any | None = None
_langfuse_checked = False


def _get_langfuse() -> Any | None:
    """Return the Langfuse client if available and enabled, else None."""
    global _langfuse, _langfuse_checked  # noqa: PLW0603
    if _langfuse_checked:
        return _langfuse
    _langfuse_checked = True
    try:
        from librarian.config import settings

        if not settings.langfuse_enabled:
            return None
        from langfuse import Langfuse  # type: ignore[import-untyped]

        _langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as exc:
        log.debug("adk.callbacks.langfuse_unavailable", error=str(exc))
        _langfuse = None
    return _langfuse


# ---------------------------------------------------------------------------
# Agent-level callbacks
# ---------------------------------------------------------------------------


async def before_agent(ctx: Context) -> types.Content | None:
    """Log when an agent starts processing."""
    agent_name = ctx.agent.name if ctx.agent else "unknown"
    session_id = ctx.session.id if ctx.session else None
    log.info("adk.agent.start", agent=agent_name, session_id=session_id)

    if ctx.session:
        ctx.session.state[f"_agent_start_{agent_name}"] = time.perf_counter()

    # Start Langfuse trace for the agent invocation
    lf = _get_langfuse()
    if lf and ctx.session:
        try:
            trace = lf.trace(
                name=f"adk.{agent_name}",
                session_id=session_id,
                metadata={"agent": agent_name, "framework": "google-adk"},
            )
            ctx.session.state[f"_lf_trace_{agent_name}"] = trace
        except Exception:
            pass

    return None


async def after_agent(ctx: Context) -> types.Content | None:
    """Log when an agent finishes processing."""
    agent_name = ctx.agent.name if ctx.agent else "unknown"
    start = ctx.session.state.get(f"_agent_start_{agent_name}") if ctx.session else None
    latency_ms = (time.perf_counter() - start) * 1000 if start else 0.0

    log.info(
        "adk.agent.done",
        agent=agent_name,
        latency_ms=round(latency_ms, 1),
        session_id=ctx.session.id if ctx.session else None,
    )

    # Flush Langfuse
    lf = _get_langfuse()
    if lf:
        try:
            lf.flush()
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Tool-level callbacks
# ---------------------------------------------------------------------------


async def before_tool(
    tool: BaseTool,
    args: dict[str, Any],
    ctx: Context,
) -> dict | None:
    """Log tool invocation with arguments."""
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    agent_name = ctx.agent.name if ctx.agent else None
    log_args = {
        k: str(v)[:100] if isinstance(v, (str, list)) else v for k, v in args.items()
    }
    log.info("adk.tool.start", tool=tool_name, args=log_args, agent=agent_name)

    if ctx.session:
        ctx.session.state[f"_tool_start_{tool_name}"] = time.perf_counter()

        # Create Langfuse span under the agent trace
        trace = ctx.session.state.get(f"_lf_trace_{agent_name}")
        if trace:
            try:
                span = trace.span(
                    name=f"tool.{tool_name}",
                    input=args,
                    metadata={"agent": agent_name},
                )
                ctx.session.state[f"_lf_span_{tool_name}"] = span
            except Exception:
                pass

    return None


async def after_tool(
    tool: BaseTool,
    args: dict[str, Any],
    ctx: Context,
    result: dict,
) -> dict | None:
    """Log tool completion with result summary."""
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    agent_name = ctx.agent.name if ctx.agent else None
    start = ctx.session.state.get(f"_tool_start_{tool_name}") if ctx.session else None
    latency_ms = (time.perf_counter() - start) * 1000 if start else 0.0

    # Build structured summary for logging
    summary: dict[str, Any] = {"latency_ms": round(latency_ms, 1)}
    if isinstance(result, dict):
        for key in ("total", "confidence", "intent", "escalated", "reason"):
            if key in result:
                summary[key] = result[key]
        if "results" in result and isinstance(result["results"], list):
            summary["result_count"] = len(result["results"])
        if "standalone_query" in result:
            summary["was_rewritten"] = result.get("was_rewritten", False)

    log.info("adk.tool.done", tool=tool_name, **summary, agent=agent_name)

    # Close Langfuse span
    if ctx.session:
        span = ctx.session.state.get(f"_lf_span_{tool_name}")
        if span:
            try:
                span.end(output=summary)
            except Exception:
                pass

    return None
