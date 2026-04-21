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

from shared.model_factory import resolve_chat_model
from ..state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """You are a routing classifier for a Billy accounting assistant.

Given the user's latest message classify the PRIMARY intent into EXACTLY ONE of:
  invoice, quote, customer, product, email, invitation, insights, expense, banking, accounting, support, direct, escalation, memory

Rules:
- "direct" only for greetings (hi, hello) or requests completely outside Billy.
- Any how-to / explanatory question → support, even if it mentions an action.
- "insights" for analytics, KPIs, dashboards, trends, aging reports, DSO, revenue summaries,
  top customers/products, overdue rates, and conversion stats. NOT for listing individual invoices.
- "expense" for logging, viewing, listing, or analysing business expenses, vendor spend,
  cost categories, fixed vs variable costs, or gross margin questions.
- "banking" for bank balance queries, bank transactions, reconciling payments to invoices,
  cashflow forecasting, or runway / burn rate questions.
- "accounting" for VAT/moms questions, audit readiness, period P&L summaries for accountants,
  unreconciled transactions (as an audit/accounting concern), or generating handoff documents.
- "escalation" when the user explicitly wants a human: "speak to a human", "talk to support",
  "this isn't working", or expresses severe frustration after multiple failed attempts.
- "memory" ONLY when the user explicitly wants to save or delete a preference: "remember that...",
  "don't forget...", "forget my ... preference", "forget everything you know about me".
- When in doubt → support.

Respond with a JSON object:
{"intent": "<one of the above>", "confidence": <0.0–1.0>}
"""


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
