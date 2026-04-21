"""Root agent (va_assistant) — routes every user message to the correct domain expert."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

import shared.memory as memory_store

from .sub_agents.accounting_agent import accounting_agent
from .sub_agents.banking_agent import banking_agent
from .sub_agents.customer_agent import customer_agent
from .sub_agents.email_agent import email_agent
from .sub_agents.expense_agent import expense_agent
from .sub_agents.insights_agent import insights_agent
from .sub_agents.invitation_agent import invitation_agent
from .sub_agents.invoice_agent import invoice_agent
from .sub_agents.product_agent import product_agent
from .sub_agents.quote_agent import quote_agent
from .sub_agents.support_agent import support_agent

logger = logging.getLogger(__name__)

_PROMPTS = Path(__file__).parent / "prompts"
_INSTRUCTION = (_PROMPTS / "va_assistant.txt").read_text()
_TRIED_AGENTS_TEMPLATE = (_PROMPTS / "router_tried_agents.txt").read_text()

# ---------------------------------------------------------------------------
# Guardrail patterns (ported from LangGraph guardrail_node)
# ---------------------------------------------------------------------------

_INJECTION_RE = re.compile(
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions"
    r"|forget\s+everything"
    r"|you\s+are\s+now\s+(a\s+)?(?!Billy|an?\s+accounting)"
    r"|system\s*:\s*you\s+are",
    re.IGNORECASE,
)

_ESCALATION_RE = re.compile(
    r"speak\s+to\s+a\s+human"
    r"|talk\s+to\s+(a\s+)?support"
    r"|this\s+isn'?t\s+working"
    r"|connect\s+me\s+(with|to)\s+(a\s+)?(human|person|agent|support)",
    re.IGNORECASE,
)

_ESCALATION_RESPONSE = (
    '{"message": "I\'ll connect you with a human supporter right away. '
    'Please hold — someone from the Billy support team will be with you shortly.", '
    '"contact_support": true}'
)

_BLOCKED_RESPONSE = (
    '{"message": "I detected an unusual pattern in your message and cannot process it. '
    'Please rephrase your request.", "contact_support": true}'
)


# ---------------------------------------------------------------------------
# Router callbacks
# ---------------------------------------------------------------------------


def provide_router_instruction(ctx: ReadonlyContext) -> str:
    state = ctx._invocation_context.session.state
    tried = state.get("tried_agents", [])
    prefs: list[dict] = state.get("user_preferences", [])

    parts: list[str] = []
    if tried:
        parts.append(_TRIED_AGENTS_TEMPLATE.format(agents=", ".join(tried)))
    if prefs:
        pref_lines = "\n".join(f"  - {p['key']}: {p['value']}" for p in prefs)
        parts.append(f"User preferences (apply these when relevant):\n{pref_lines}")

    return "\n\n".join(parts)


async def _before_agent_callback(callback_context: CallbackContext) -> types.Content | None:
    """Clear tried_agents once per invocation. Load user preferences on first turn."""
    invocation_id = callback_context._invocation_context.invocation_id
    if callback_context.state.get("_tried_agents_invocation") != invocation_id:
        callback_context.state["tried_agents"] = []
        callback_context.state["_tried_agents_invocation"] = invocation_id

    # Load preferences from memory store on the first turn of this session
    if "user_preferences" not in callback_context.state:
        user_id = callback_context.state.get("user_id", "default")
        try:
            prefs = await memory_store.get_top(user_id)
            callback_context.state["user_preferences"] = prefs
        except Exception as e:
            logger.warning("Could not load user preferences: %s", e)
            callback_context.state["user_preferences"] = []

    return None


def _guardrail_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Injection detection + escalation trigger check on the latest user content.

    Runs before every model call on the root router.  Returns an LlmResponse
    to short-circuit the model when a pattern is detected; returns None to allow
    the call to proceed normally.
    """
    if not llm_request.contents:
        return None

    for content in reversed(llm_request.contents):
        role = getattr(content, "role", "")
        if role != "user":
            continue
        parts = getattr(content, "parts", [])
        text = "".join(getattr(p, "text", "") or "" for p in parts)

        if _ESCALATION_RE.search(text):
            logger.info("Escalation trigger detected — routing to human supporter")
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=_ESCALATION_RESPONSE)],
                )
            )

        if _INJECTION_RE.search(text):
            logger.warning("Injection pattern detected in user message")
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=_BLOCKED_RESPONSE)],
                )
            )
        break  # only check the most recent user turn

    return None


# ---------------------------------------------------------------------------
# Memory tools — exposed on the root agent so any turn can trigger them
# ---------------------------------------------------------------------------


async def update_user_preference(key: str, value: str, tool_context: Any = None) -> dict:
    """Remember a user preference. Call when the user says 'remember that...' or 'don't forget...'."""
    user_id = "default"
    try:
        if tool_context is not None:
            user_id = tool_context.state.get("user_id", "default")
    except Exception:
        pass
    await memory_store.upsert(user_id, f"pref:{key}", value)
    try:
        if tool_context is not None:
            tool_context.state["user_preferences"] = await memory_store.get_top(user_id)
    except Exception:
        pass
    return {"success": True, "message": f"I'll remember that {key} is {value}."}


async def delete_user_preference(key: str, tool_context: Any = None) -> dict:
    """Forget a stored user preference. Call when the user says 'forget my ... preference'."""
    user_id = "default"
    try:
        if tool_context is not None:
            user_id = tool_context.state.get("user_id", "default")
    except Exception:
        pass
    await memory_store.delete(user_id, f"pref:{key}")
    try:
        if tool_context is not None:
            tool_context.state["user_preferences"] = await memory_store.get_top(user_id)
    except Exception:
        pass
    return {"success": True, "message": f"I've forgotten your {key} preference."}


# ---------------------------------------------------------------------------
# Root agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    model="gemini-2.5-flash-lite",
    name="va_assistant",
    description=(
        "Routing assistant for the Billy accounting platform. Classifies user requests "
        "and delegates to the correct domain expert: invoices, quotes, customers, products, "
        "emails, invitations, insights, expenses, banking, accounting, or support. "
        "Does not answer domain questions directly."
    ),
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=150,
    ),
    static_instruction=types.Content(
        role="user",
        parts=[types.Part(text=_INSTRUCTION)],
    ),
    instruction=provide_router_instruction,
    tools=[update_user_preference, delete_user_preference],
    sub_agents=[
        invoice_agent,
        quote_agent,
        customer_agent,
        product_agent,
        email_agent,
        invitation_agent,
        insights_agent,
        expense_agent,
        banking_agent,
        accounting_agent,
        support_agent,
    ],
    before_agent_callback=_before_agent_callback,
    before_model_callback=_guardrail_callback,
)
