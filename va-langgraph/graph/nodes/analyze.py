"""Analyze node — classifies intent and extracts key entities.

Uses a small, fast LLM call (Haiku-equivalent) to classify the user's intent
into one of the routing buckets.  The classification result drives the
conditional edge in the graph.

Intent values:
  invoice    — create, view, list, edit, void, or remind on invoices
  quote      — create, view, list, edit quotes; convert to invoice
  customer   — manage customers / contacts
  product    — manage products / services
  email      — send invoice or quote by email
  invitation — invite a user to the organisation
  insights   — KPI dashboards, revenue trends, aging, DSO, customer/product analytics
  expense    — log, view, list, or analyse business expenses; vendor spend; gross margin
  banking    — bank balances, transactions, reconciliation, cashflow forecast, runway
  accounting — VAT (moms) reporting, audit readiness, period P&L, handoff docs
  support    — how-to questions, explanations, fallback
  direct     — greeting or completely out-of-scope (answer inline, no domain)
  escalation — user explicitly wants a human agent
  memory     — user explicitly wants to remember or forget a preference
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from model_factory import resolve_chat_model
from ..state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = (Path(__file__).parent.parent.parent / "prompts" / "router.txt").read_text()


async def analyze_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    if not messages:
        return {**state, "intent": "direct", "routing_confidence": 1.0}

    user_text = messages[-1].content

    # Include page URL as context if available
    page_url = state.get("page_url")
    if page_url:
        user_text = f"[User is on page: {page_url}]\n{user_text}"

    # Inject user preferences so the classifier can use them
    user_preferences = state.get("user_preferences", [])
    if user_preferences:
        pref_lines = "; ".join(f"{p['key']}={p['value']}" for p in user_preferences)
        user_text = f"[User preferences: {pref_lines}]\n{user_text}"

    try:
        resp = await resolve_chat_model("small").ainvoke([
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
