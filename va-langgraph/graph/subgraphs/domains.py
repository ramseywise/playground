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

from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

import shared.artefact_store as artefact_store
from shared.model_factory import resolve_chat_model
from ..state import AgentState
from .base import run_domain

logger = logging.getLogger(__name__)

_BILLY_MCP_URL = os.getenv("BILLY_MCP_URL", "http://localhost:8765/sse")

_BILLY_SERVER = {"billy": {"url": _BILLY_MCP_URL, "transport": "sse"}}

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
# System prompts (concise — format_node does the full rendering)
# ---------------------------------------------------------------------------

_INVOICE_SYSTEM = """You are Billy, an accounting assistant. Handle invoice operations.
Use list_customers / list_products to resolve names to IDs before creating.
Call tools in parallel when both are needed. VAT is 25%. Default currency DKK, net 7 days."""

_QUOTE_SYSTEM = """You are Billy, an accounting assistant. Handle quote operations.
Use list_customers / list_products to resolve names. VAT 25%. Default expiry 30 days."""

_CUSTOMER_SYSTEM = """You are Billy, an accounting assistant. Handle customer/contact management.
Default country DK. CVR is the Danish company registration number."""

_PRODUCT_SYSTEM = """You are Billy, an accounting assistant. Handle product and service management.
Prices are always excl. VAT."""

_EMAIL_SYSTEM = """You are Billy, an accounting assistant. Send invoices and quotes by email.
Draft a professional Danish subject and body if not provided by the user."""

_INVITATION_SYSTEM = """You are Billy, an accounting assistant. Invite users by email.
Invited users receive the collaborator role."""

_EXPENSE_SYSTEM = """You are Billy, an accounting assistant. Handle expense tracking and analysis.
Amounts are always excl. VAT. Danish VAT (moms) is 25%; rent and salaries are VAT-exempt (0%).
Categories: rent, salaries, software, marketing, office, travel, meals, professional_services, utilities, other.
Set is_fixed=true for recurring fixed costs. Call tools in parallel for multi-metric queries."""

_BANKING_SYSTEM = """You are Billy, an accounting assistant. Handle banking and cashflow analysis.
Use get_bank_balance to show current balances. Use list_bank_transactions for transaction history.
Use match_transaction_to_invoice to reconcile a payment with an invoice.
Use get_cashflow_forecast to project inflow/outflow for the next N months.
Use get_runway_estimate to show months of runway at current burn rate.
Runway = total bank balance ÷ average monthly expenses. Amounts in DKK."""

_ACCOUNTING_SYSTEM = """You are Billy, an accounting assistant. Handle VAT reporting, audit readiness, and P&L summaries.
Danish VAT (moms) is 25%. Reporting is quarterly for most companies.
Output VAT = salgsmoms (collected on sales). Input VAT = købsmoms (paid on purchases).
Net payable = output_vat - input_vat. Negative = SKAT refund.
Quarterly deadlines: Q1→1 Jun, Q2→1 Sep, Q3→1 Dec, Q4→1 Mar following year.
Use get_vat_summary for VAT questions. Use get_period_summary for P&L.
Use get_audit_readiness_score for audit preparation. Use get_unreconciled_transactions for reconciliation gaps.
Use generate_handoff_doc to produce a markdown document for accountant handoff.
After generate_handoff_doc, always call save_artefact with the returned markdown_summary as content
and filename like 'handoff_<period>.md'. Include artefact_id and artefact_url in your response."""

_SUPPORT_SYSTEM = """You are Billy, an accounting assistant. Answer how-to questions about Billy
by searching the official help docs. Pass 2-3 Danish search terms. Reference source URLs."""

_INSIGHTS_SYSTEM = """You are Billy, an accounting assistant. Answer analytics and KPI questions.
Use get_insight_revenue_summary for revenue KPI cards (totals, YoY delta).
Use get_insight_invoice_status for status breakdown (draft/unpaid/paid/overdue).
Use get_insight_monthly_revenue for monthly trend charts.
Use get_insight_top_customers for customer revenue ranking.
Use get_insight_aging_report for AR aging buckets. Filter by customer if named.
Use get_insight_customer_summary for per-customer KPIs + open invoices.
Use get_insight_product_revenue for product revenue ranking.
Use get_invoice_lines_summary for revenue per product from invoice lines.
Use get_invoice_dso_stats for average payment speed and overdue rate.
Cross-domain tools (require both invoice and expense data):
Use get_net_margin for net margin: revenue minus all costs. Pass year or period.
Use get_margin_by_product for per-product margin (COGS allocated proportionally).
Use get_customer_concentration for concentration risk: top-1%, top-3%, HHI.
Use get_dso_trend for monthly DSO trend over N months.
Use get_break_even_estimate for monthly fixed costs, variable rate, break-even revenue.
Use detect_anomaly to flag monthly metric outliers (revenue/expenses/overdue_rate/dso).
Call tools in parallel when answering multi-metric questions.
Present amounts in DKK with VAT status noted."""

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
