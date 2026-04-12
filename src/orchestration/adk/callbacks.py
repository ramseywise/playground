"""ADK agent and tool callbacks for observability.

Provides structured logging for every agent invocation and tool call,
so ADK agents have the same observability as the LangGraph pipeline.
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
# Agent-level callbacks
# ---------------------------------------------------------------------------


async def before_agent(ctx: Context) -> types.Content | None:
    """Log when an agent starts processing."""
    agent_name = ctx.agent.name if ctx.agent else "unknown"
    log.info(
        "adk.agent.start",
        agent=agent_name,
        session_id=ctx.session.id if ctx.session else None,
    )
    # Store start time in state for latency calculation
    ctx.session.state[f"_agent_start_{agent_name}"] = time.perf_counter()
    return None


async def after_agent(ctx: Context) -> types.Content | None:
    """Log when an agent finishes processing."""
    agent_name = ctx.agent.name if ctx.agent else "unknown"
    start = ctx.session.state.get(f"_agent_start_{agent_name}")
    latency_ms = (time.perf_counter() - start) * 1000 if start else 0.0

    log.info(
        "adk.agent.done",
        agent=agent_name,
        latency_ms=round(latency_ms, 1),
        session_id=ctx.session.id if ctx.session else None,
    )
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
    # Truncate large args for logging
    log_args = {
        k: str(v)[:100] if isinstance(v, (str, list)) else v for k, v in args.items()
    }
    log.info(
        "adk.tool.start",
        tool=tool_name,
        args=log_args,
        agent=ctx.agent.name if ctx.agent else None,
    )
    ctx.session.state[f"_tool_start_{tool_name}"] = time.perf_counter()
    return None  # Don't override — let the tool run normally


async def after_tool(
    tool: BaseTool,
    args: dict[str, Any],
    ctx: Context,
    result: dict,
) -> dict | None:
    """Log tool completion with result summary."""
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    start = ctx.session.state.get(f"_tool_start_{tool_name}")
    latency_ms = (time.perf_counter() - start) * 1000 if start else 0.0

    # Summarize result for logging (don't log full text content)
    summary: dict[str, Any] = {"latency_ms": round(latency_ms, 1)}
    if isinstance(result, dict):
        if "total" in result:
            summary["total"] = result["total"]
        if "confidence" in result:
            summary["confidence"] = result["confidence"]
        if "results" in result and isinstance(result["results"], list):
            summary["result_count"] = len(result["results"])
        if "intent" in result:
            summary["intent"] = result["intent"]
        if "standalone_query" in result:
            summary["was_rewritten"] = result.get("was_rewritten", False)

    log.info(
        "adk.tool.done",
        tool=tool_name,
        **summary,
        agent=ctx.agent.name if ctx.agent else None,
    )
    return None  # Don't modify the result
