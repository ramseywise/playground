"""Domain subgraph nodes — one per intent bucket.

Each function opens a short-lived MCP connection to Billy, fetches the domain
tools, and runs the ReAct loop.

support_subgraph calls the hc-rag-agent HTTP service (HC_RAG_AGENT_URL).
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import structlog
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

import artefact_store as artefact_store
from ..state import AgentState
from .base import run_domain

_HC_RAG_URL = os.getenv("HC_RAG_AGENT_URL", "http://localhost:8002")

log = structlog.get_logger(__name__)

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

_INVOICE_TOOLS = frozenset(
    [
        "get_invoice",
        "list_invoices",
        "get_invoice_summary",
        "get_invoice_dso_stats",
        "list_customers",
        "list_products",
    ]
)

_QUOTE_TOOLS = frozenset(
    [
        "list_quotes",
        "get_quote_conversion_stats",
        "list_customers",
        "list_products",
    ]
)

_CUSTOMER_TOOLS = frozenset(["list_customers", "get_customer"])

_PRODUCT_TOOLS = frozenset(["list_products", "get_product"])

_EMAIL_TOOLS = frozenset(
    []
)  # TODO(2): add send_invoice_by_email, send_quote_by_email (need test account)

_INVITATION_TOOLS = frozenset([])  # TODO(2): add invite_user (need test account)

_EXPENSE_TOOLS = frozenset(
    [
        "list_expenses",
        "get_expense",
        "get_expense_summary",
        "get_vendor_spend",
        "get_expenses_by_category",
        "get_gross_margin",
    ]
)

_BANKING_TOOLS = frozenset(
    [
        "get_bank_balance",
        "list_bank_transactions",
        "get_cashflow_forecast",
        "get_runway_estimate",
        # TODO(2): add match_transaction_to_invoice (need test account)
    ]
)

_ACCOUNTING_TOOLS = frozenset(
    [
        "get_vat_summary",
        "get_unreconciled_transactions",
        "get_audit_readiness_score",
        "get_period_summary",
        "generate_handoff_doc",
    ]
)

_INSIGHTS_TOOLS = frozenset(
    [
        "get_insight_revenue_summary",
        "get_insight_invoice_status",
        "get_insight_monthly_revenue",
        "get_insight_top_customers",
        "get_insight_aging_report",
        "get_insight_customer_summary",
        "get_insight_product_revenue",
        "get_invoice_lines_summary",
        "get_invoice_dso_stats",
        "get_net_margin",
        "get_margin_by_product",
        "get_customer_concentration",
        "get_dso_trend",
        "get_break_even_estimate",
        "detect_anomaly",
    ]
)

# ---------------------------------------------------------------------------
# System prompts — loaded from prompts/ at import time
# ---------------------------------------------------------------------------

_INVOICE_SYSTEM = (_PROMPTS / "invoice.txt").read_text()
_QUOTE_SYSTEM = (_PROMPTS / "quote.txt").read_text()
_CUSTOMER_SYSTEM = (_PROMPTS / "customer.txt").read_text()
_PRODUCT_SYSTEM = (_PROMPTS / "product.txt").read_text()
_EMAIL_SYSTEM = (_PROMPTS / "email.txt").read_text()
_INVITATION_SYSTEM = (_PROMPTS / "invitation.txt").read_text()
_EXPENSE_SYSTEM = (_PROMPTS / "expense.txt").read_text()
_BANKING_SYSTEM = (_PROMPTS / "banking.txt").read_text()
_ACCOUNTING_SYSTEM = (_PROMPTS / "accounting.txt").read_text()
_INSIGHTS_SYSTEM = (_PROMPTS / "insights.txt").read_text()

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
    if not _EMAIL_TOOLS:
        tool_results = list(state.get("tool_results", []))
        tool_results.append(
            {
                "tool": "email",
                "result": "Email sending is not yet available. Please send invoices manually from the Billy dashboard.",
            }
        )
        return {**state, "tool_results": tool_results}
    async with MultiServerMCPClient(_BILLY_SERVER) as client:
        tools = _filter(client.get_tools(), _EMAIL_TOOLS)
        return await run_domain(state, _EMAIL_SYSTEM, tools)


async def invitation_subgraph(state: AgentState) -> AgentState:
    if not _INVITATION_TOOLS:
        tool_results = list(state.get("tool_results", []))
        tool_results.append(
            {
                "tool": "invitation",
                "result": "User invitations are not yet available. Please manage team members directly in Billy Settings.",
            }
        )
        return {**state, "tool_results": tool_results}
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
    """Retrieve support docs via hc-rag-agent HTTP service.

    Calls retrieval-only endpoint (/api/v1/retrieval) which returns structured documents.
    va-langgraph then synthesizes the answer using its own LLM.
    """
    user_text = str(state["messages"][-1].content)
    page_url = state.get("page_url")
    query = f"[User is on page: {page_url}]\n{user_text}" if page_url else user_text

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{_HC_RAG_URL}/api/v1/retrieval",
            json={"thread_id": state.get("session_id", "default"), "query": query},
        )
        r.raise_for_status()

    result = r.json()
    documents = result.get("documents") or []
    escalated = result.get("escalated", False)
    confidence = result.get("confidence_score", 0.0)

    tool_results = list(state.get("tool_results", []))
    tool_results.append(
        {
            "tool": "support_retrieval",
            "args": {"query": query},
            "result": {
                "documents": documents,
                "confidence": confidence,
                "escalated": escalated,
            },
        }
    )
    return {**state, "tool_results": tool_results}


async def insights_subgraph(state: AgentState) -> AgentState:
    async with MultiServerMCPClient(_BILLY_SERVER) as client:
        tools = _filter(client.get_tools(), _INSIGHTS_TOOLS)
        return await run_domain(state, _INSIGHTS_SYSTEM, tools)


async def accounting_subgraph(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "default")

    @tool
    async def save_artefact(
        content: str, filename: str, content_type: str = "text/markdown"
    ) -> dict:
        """Save generated content as a downloadable artefact. Call after generate_handoff_doc
        to make the document downloadable. Returns artefact_id and url."""
        return await artefact_store.save(session_id, content, filename, content_type)

    async with MultiServerMCPClient(_BILLY_SERVER) as client:
        tools = _filter(client.get_tools(), _ACCOUNTING_TOOLS) + [save_artefact]
        return await run_domain(state, _ACCOUNTING_SYSTEM, tools)
