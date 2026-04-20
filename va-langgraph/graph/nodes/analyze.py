"""Analyze node — classifies intent and extracts key entities.

Uses a small, fast LLM call (Haiku-equivalent) to classify the user's intent
into one of the routing buckets.  The classification result drives the
conditional edge in the graph.

Intent values:
  invoice    — create, view, list, edit, approve invoices
  quote      — create, view, list quotes; convert to invoice
  customer   — manage customers / contacts
  product    — manage products / services
  email      — send invoice or quote by email
  invitation — invite a user to the organisation
  support    — how-to questions, explanations, fallback
  direct     — greeting or completely out-of-scope (answer inline, no domain)
"""

from __future__ import annotations

import json
import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from ..state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """You are a routing classifier for a Billy accounting assistant.

Given the user's latest message classify the PRIMARY intent into EXACTLY ONE of:
  invoice, quote, customer, product, email, invitation, support, direct

Rules:
- "direct" only for greetings (hi, hello) or requests completely outside Billy.
- Any how-to / explanatory question → support, even if it mentions an action.
- When in doubt → support.

Respond with a JSON object:
{"intent": "<one of the above>", "confidence": <0.0–1.0>}
"""

def _get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)


async def analyze_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    if not messages:
        return {**state, "intent": "direct", "routing_confidence": 1.0}

    user_text = messages[-1].content

    # Include page URL as context if available
    page_url = state.get("page_url")
    if page_url:
        user_text = f"[User is on page: {page_url}]\n{user_text}"

    try:
        resp = await _get_llm().ainvoke([
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": str(user_text)},
        ])
        raw = resp.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        parsed = json.loads(raw)
        intent = parsed.get("intent", "support")
        confidence = float(parsed.get("confidence", 0.8))
    except Exception as e:
        logger.warning("analyze_node failed: %s — defaulting to support", e)
        intent = "support"
        confidence = 0.5

    return {**state, "intent": intent, "routing_confidence": confidence}
