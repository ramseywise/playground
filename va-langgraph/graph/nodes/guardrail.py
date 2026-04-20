"""Guardrail node — runs before any LLM call.

Checks:
  1. Message size (hard cap at 4 000 chars — truncates with notice)
  2. Prompt injection heuristics
  3. PII redaction (emails → [EMAIL], phones → [PHONE])

Sets state["blocked"] = True to short-circuit the graph on a hard refuse.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

from ..state import AgentState

_MAX_CHARS = 4_000

# Injection patterns — instruction-override and jailbreak heuristics
_INJECTION_RE = re.compile(
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions"
    r"|forget\s+everything"
    r"|you\s+are\s+now\s+(a\s+)?(?!Billy|an?\s+accounting)"
    r"|system\s*:\s*you\s+are",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+45\s?)?\d[\d\s\-]{6,}\d\b")


def guardrail_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    if not messages:
        return state

    last = messages[-1]
    text: str = last.content if isinstance(last.content, str) else str(last.content)

    # 1. Size check — truncate oversize messages
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + f"\n\n[Message truncated to {_MAX_CHARS} characters]"

    # 2. Injection detection
    if _INJECTION_RE.search(text):
        return {
            **state,
            "blocked": True,
            "block_reason": "Message contains patterns that look like prompt injection.",
        }

    # 3. PII redaction
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)

    # Replace the last message with the cleaned text
    cleaned_messages = list(messages[:-1]) + [HumanMessage(content=text)]

    return {
        **state,
        "messages": cleaned_messages,
        "blocked": False,
        "block_reason": None,
    }
