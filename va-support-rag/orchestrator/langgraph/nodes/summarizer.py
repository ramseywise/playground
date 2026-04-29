"""Summarization node (LangGraph id: ``summarizer``).

Fires when ``len(state.messages) >= RAG_SUMMARIZATION_THRESHOLD`` (default 8).
Uses a cheap model (Haiku / Flash) to compress conversation history into a
SystemMessage summary, then replaces messages with:

    [SystemMessage("Conversation summary: ..."), ...last RAG_SUMMARIZATION_KEEP messages]

The node is gated behind ``route_to_summarizer`` so it only fires on the
answer→END edge when the threshold is reached.

Config env vars
---------------
RAG_SUMMARIZATION_ENABLED      true | false  (default: true)
RAG_SUMMARIZATION_THRESHOLD    int           (default: 8)
RAG_SUMMARIZATION_KEEP         int           (default: 4)
"""

from __future__ import annotations

import logging
import time
from typing import Any


from orchestrator.langgraph.schemas.state import GraphState
from orchestrator.history import messages_after_summary, should_summarize

log = logging.getLogger(__name__)


def _get_summarizer_chain() -> Any:
    """Return a cheap LLM chain for summarization — lazy import to avoid circular deps."""
    from orchestrator.langgraph.chains import get_summarizer_chain  # noqa: PLC0415

    return get_summarizer_chain()


def summarizer_node(state: GraphState) -> dict:
    """Compress state.messages and return updated messages list."""
    from core.config import (
        RAG_SUMMARIZATION_ENABLED,
        RAG_SUMMARIZATION_KEEP,
        RAG_SUMMARIZATION_THRESHOLD,
    )  # noqa: PLC0415

    if not RAG_SUMMARIZATION_ENABLED:
        return {}

    msgs = list(state.messages)
    if not should_summarize(msgs, threshold=RAG_SUMMARIZATION_THRESHOLD):
        return {}

    log.info("summarizer: start message_count=%d", len(msgs))
    t0 = time.perf_counter()

    # Build a plain-text transcript for the summarizer LLM.
    lines: list[str] = []
    for m in msgs:
        role = type(m).__name__.replace("Message", "")
        content = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {content[:800]}")
    transcript = "\n".join(lines)

    try:
        chain = _get_summarizer_chain()
        result = chain.invoke({"transcript": transcript})
        summary_text = result.content if hasattr(result, "content") else str(result)
    except Exception:
        log.exception("summarizer: llm_failed skipping_summarization")
        return {}

    elapsed_ms = (time.perf_counter() - t0) * 1000
    new_messages = messages_after_summary(
        summary_text,
        msgs,
        keep_recent=RAG_SUMMARIZATION_KEEP,
    )
    log.info(
        "summarizer: done original=%d compressed=%d elapsed_ms=%.1f",
        len(msgs),
        len(new_messages),
        elapsed_ms,
    )
    return {
        "messages": new_messages,
        "latency_ms": {**state.latency_ms, "summarizer_ms": elapsed_ms},
    }


__all__ = ["summarizer_node"]
