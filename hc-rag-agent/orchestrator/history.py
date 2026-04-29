"""Conversation history utilities: pruning and summarization support.

Pruning
-------
``prune_messages`` caps the messages list to the N most recent turns (default 20),
keeping the first message (original question anchor) so the planner always has
context. Call this at the top of any node that reads ``state.messages``.

Summarization trigger
---------------------
``should_summarize`` returns True when the message count crosses the threshold
(default 8) — used as a conditional edge guard before the ``summarizer`` node.
"""

from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.messages import AnyMessage, SystemMessage

log = logging.getLogger(__name__)

_DEFAULT_MAX_MESSAGES = 20
_DEFAULT_SUMMARIZE_THRESHOLD = 8
_DEFAULT_KEEP_RECENT = 4


def prune_messages(
    messages: Sequence[AnyMessage],
    *,
    max_messages: int = _DEFAULT_MAX_MESSAGES,
) -> list[AnyMessage]:
    """Return at most *max_messages* messages, always keeping the first.

    When truncation happens the first message (anchor) is kept so the planner
    still sees the original question, then the most recent ``max_messages - 1``
    messages follow.
    """
    msgs = list(messages)
    if len(msgs) <= max_messages:
        return msgs
    log.info(
        "history.prune original=%d max=%d kept=%d",
        len(msgs),
        max_messages,
        max_messages,
    )
    return [msgs[0]] + msgs[-(max_messages - 1) :]


def should_summarize(
    messages: Sequence[AnyMessage],
    *,
    threshold: int = _DEFAULT_SUMMARIZE_THRESHOLD,
) -> bool:
    """Return True when the message list is long enough to warrant summarization."""
    return len(messages) >= threshold


def messages_after_summary(
    summary_text: str,
    recent_messages: Sequence[AnyMessage],
    *,
    keep_recent: int = _DEFAULT_KEEP_RECENT,
) -> list[AnyMessage]:
    """Build a pruned messages list: summary SystemMessage + last *keep_recent* messages."""
    summary_msg = SystemMessage(
        content=f"[Conversation summary]\n{summary_text}",
    )
    recent = list(recent_messages)[-keep_recent:]
    return [summary_msg] + recent


__all__ = [
    "messages_after_summary",
    "prune_messages",
    "should_summarize",
]
