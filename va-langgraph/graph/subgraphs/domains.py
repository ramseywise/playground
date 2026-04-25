"""Domain subgraph nodes — one per intent bucket.

Each function opens a short-lived MCP connection to Billy, fetches the domain
tools, runs the ReAct loop, then closes the connection.

support_subgraph uses a CRAG loop (retrieve → grade → rewrite, max 2 retries)
before handing off to format_node via tool_results.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

import artefact_store as artefact_store
from model_factory import resolve_chat_model
from ..state import AgentState
from .base import run_domain

logger = logging.getLogger(__name__)

_BILLY_MCP_URL = os.getenv("BILLY_MCP_URL", "http://localhost:8765/sse")

_BILLY_SERVER = {"billy": {"url": _BILLY_MCP_URL, "transport": "sse"}}

_PROMPTS = Path(__file__).parent.parent.parent / "prompts"

# ---------------------------------------------------------------------------
# Tool name sets per domain
# ---------------------------------------------------------------------------

# TODO(2): re-add write tools once a Billy test account is available.
# Write tools (create_*, edit_*, void_*, send_*, invite_*, match_*) require
# live Billy credentials — they work against the local SQLite stub but must
# not be exposed to users until a real test account is set up.

_INVOICE_TOOLS = frozenset([
    "get_invoice", "list_invoices", "get_invoice_summary",
    "get_invoice_dso_stats",
    "list_customers", "list_products",
])

_QUOTE_TOOLS = frozenset([
    "list_quotes", "get_quote_conversion_stats",
    "list_customers", "list_products",
])

_CUSTOMER_TOOLS = frozenset(["list_customers", "get_customer"])

_PRODUCT_TOOLS = frozenset(["list_products", "get_product"])

_EMAIL_TOOLS = frozenset([])      # TODO(2): add send_invoice_by_email, send_quote_by_email (need test account)

_INVITATION_TOOLS = frozenset([]) # TODO(2): add invite_user (need test account)

_EXPENSE_TOOLS = frozenset([
    "list_expenses", "get_expense",
    "get_expense_summary", "get_vendor_spend",
    "get_expenses_by_category", "get_gross_margin",
])

_BANKING_TOOLS = frozenset([
    "get_bank_balance", "list_bank_transactions",
    "get_cashflow_forecast", "get_runway_estimate",
    # TODO(2): add match_transaction_to_invoice (need test account)
])

_ACCOUNTING_TOOLS = frozenset([
    "get_vat_summary", "get_unreconciled_transactions",
    "get_audit_readiness_score", "get_period_summary", "generate_handoff_doc",
])

_SUPPORT_TOOLS = frozenset(["fetch_support_knowledge"])

_INSIGHTS_TOOLS = frozenset([
    "get_insight_revenue_summary", "get_insight_invoice_status",
    "get_insight_monthly_revenue", "get_insight_top_customers",
    "get_insight_aging_report", "get_insight_customer_summary",
    "get_insight_product_revenue", "get_invoice_lines_summary",
    "get_invoice_dso_stats",
    "get_net_margin", "get_margin_by_product", "get_customer_concentration",
    "get_dso_trend", "get_break_even_estimate", "detect_anomaly",
])

# ---------------------------------------------------------------------------
# System prompts — loaded from prompts/ at import time
# ---------------------------------------------------------------------------

_INVOICE_SYSTEM     = (_PROMPTS / "invoice.txt").read_text()
_QUOTE_SYSTEM       = (_PROMPTS / "quote.txt").read_text()
_CUSTOMER_SYSTEM    = (_PROMPTS / "customer.txt").read_text()
_PRODUCT_SYSTEM     = (_PROMPTS / "product.txt").read_text()
_EMAIL_SYSTEM       = (_PROMPTS / "email.txt").read_text()
_INVITATION_SYSTEM  = (_PROMPTS / "invitation.txt").read_text()
_EXPENSE_SYSTEM     = (_PROMPTS / "expense.txt").read_text()
_BANKING_SYSTEM     = (_PROMPTS / "banking.txt").read_text()
_ACCOUNTING_SYSTEM  = (_PROMPTS / "accounting.txt").read_text()
_SUPPORT_SYSTEM     = (_PROMPTS / "support.txt").read_text()
_INSIGHTS_SYSTEM    = (_PROMPTS / "insights.txt").read_text()

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _filter(all_tools, names: frozenset):
    return [t for t in all_tools if t.name in names]


# ---------------------------------------------------------------------------
# Subgraph node functions
# ---------------------------------------------------------------------------


async def invoice_subgraph(state: AgentState) -> AgentState:
    async with MultiServerMCPClient(_BILLY_SERVER) as client:
        tools = _filter(client.get_tools(), _INVOICE_TOOLS)
        return await run_domain(state, _INVOICE_SYSTEM, tools)


async def quote_subgraph(state: AgentState) -> AgentState:
    async with MultiServerMCPClient(_BILLY_SERVER) as client:
        tools = _filter(client.get_tools(), _QUOTE_TOOLS)
        return await run_domain(state, _QUOTE_SYSTEM, tools)


async def customer_subgraph(state: AgentState) -> AgentState:
    async with MultiServerMCPClient(_BILLY_SERVER) as client:
        tools = _filter(client.get_tools(), _CUSTOMER_TOOLS)
        return await run_domain(state, _CUSTOMER_SYSTEM, tools)


async def product_subgraph(state: AgentState) -> AgentState:
    async with MultiServerMCPClient(_BILLY_SERVER) as client:
        tools = _filter(client.get_tools(), _PRODUCT_TOOLS)
        return await run_domain(state, _PRODUCT_SYSTEM, tools)


async def email_subgraph(state: AgentState) -> AgentState:
    async with MultiServerMCPClient(_BILLY_SERVER) as client:
        tools = _filter(client.get_tools(), _EMAIL_TOOLS)
        return await run_domain(state, _EMAIL_SYSTEM, tools)


async def invitation_subgraph(state: AgentState) -> AgentState:
    async with MultiServerMCPClient(_BILLY_SERVER) as client:
        tools = _filter(client.get_tools(), _INVITATION_TOOLS)
        return await run_domain(state, _INVITATION_SYSTEM, tools)


async def expense_subgraph(state: AgentState) -> AgentState:
    async with MultiServerMCPClient(_BILLY_SERVER) as client:
        tools = _filter(client.get_tools(), _EXPENSE_TOOLS)
        return await run_domain(state, _EXPENSE_SYSTEM, tools)


async def banking_subgraph(state: AgentState) -> AgentState:
    async with MultiServerMCPClient(_BILLY_SERVER) as client:
        tools = _filter(client.get_tools(), _BANKING_TOOLS)
        return await run_domain(state, _BANKING_SYSTEM, tools)


async def support_subgraph(state: AgentState) -> AgentState:
    """CRAG loop: retrieve → grade → rewrite (max 2 retries).

    Calls fetch_support_knowledge up to 3 times with progressively refined
    queries, grades each result for relevance, then hands the best docs to
    format_node via tool_results.  Falls back to a plain run_domain pass if
    the MCP tool is unavailable.
    """
    async with MultiServerMCPClient(_BILLY_SERVER) as client:
        support_tools = _filter(client.get_tools(), _SUPPORT_TOOLS)
        fetch_tool = next((t for t in support_tools if t.name == "fetch_support_knowledge"), None)

        if fetch_tool is None:
            return await run_domain(state, _SUPPORT_SYSTEM, support_tools)

        user_text = str(state["messages"][-1].content)
        page_url = state.get("page_url")
        if page_url:
            user_text = f"[User is on page: {page_url}]\n{user_text}"

        query = user_text
        best_docs = None
        grader = resolve_chat_model("small")

        for attempt in range(3):  # initial + up to 2 rewrites
            docs = await fetch_tool.ainvoke({"query": query})
            best_docs = docs

            if attempt >= 2:
                break  # used all retries — accept whatever we have

            grade_prompt = (
                f'User question: "{user_text}"\n\n'
                f'Retrieved docs (truncated): {json.dumps(docs, default=str)[:1500]}\n\n'
                f'Are these docs sufficient to answer the question?\n'
                f'Respond with JSON only: {{"sufficient": true/false, '
                f'"rewritten_query": "better search query if not sufficient, else empty string"}}'
            )
            try:
                resp = await grader.ainvoke(grade_prompt)
                raw = resp.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1].strip().lstrip("json").strip()
                graded = json.loads(raw)
                if graded.get("sufficient", True):
                    break
                new_q = graded.get("rewritten_query", "").strip()
                if new_q:
                    query = new_q
            except Exception as e:
                logger.debug("CRAG grader failed (attempt %d): %s", attempt, e)
                break  # grading failed — use best_docs as-is

        tool_results = list(state.get("tool_results", []))
        tool_results.append({
            "tool": "fetch_support_knowledge",
            "args": {"query": query},
            "result": best_docs,
        })
        return {**state, "tool_results": tool_results}


async def insights_subgraph(state: AgentState) -> AgentState:
    async with MultiServerMCPClient(_BILLY_SERVER) as client:
        tools = _filter(client.get_tools(), _INSIGHTS_TOOLS)
        return await run_domain(state, _INSIGHTS_SYSTEM, tools)


async def accounting_subgraph(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "default")

    @tool
    async def save_artefact(content: str, filename: str, content_type: str = "text/markdown") -> dict:
        """Save generated content as a downloadable artefact. Call after generate_handoff_doc
        to make the document downloadable. Returns artefact_id and url."""
        return await artefact_store.save(session_id, content, filename, content_type)

    async with MultiServerMCPClient(_BILLY_SERVER) as client:
        tools = _filter(client.get_tools(), _ACCOUNTING_TOOLS) + [save_artefact]
        return await run_domain(state, _ACCOUNTING_SYSTEM, tools)
