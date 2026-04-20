"""Domain subgraph nodes — one per intent bucket.

Each function receives AgentState, runs the domain ReAct loop,
and returns updated state with tool_results populated.
"""

from __future__ import annotations

from langchain_core.tools import tool as lc_tool

from shared.tools.customers import create_customer, edit_customer, list_customers
from shared.tools.emails import send_invoice_by_email, send_quote_by_email
from shared.tools.invitations import invite_user
from shared.tools.invoices import (
    create_invoice,
    edit_invoice,
    get_invoice,
    get_invoice_summary,
    list_invoices,
)
from shared.tools.products import create_product, edit_product, list_products
from shared.tools.quotes import (
    create_invoice_from_quote,
    create_quote,
    list_quotes,
)
from shared.tools.support_knowledge import fetch_support_knowledge
from ..state import AgentState
from .base import run_domain

# ---------------------------------------------------------------------------
# Wrap plain Python functions as LangChain tools
# ---------------------------------------------------------------------------

def _t(fn):
    """Wrap a plain function as a LangChain @tool preserving its docstring."""
    return lc_tool(fn)


_invoice_tools = [_t(f) for f in [
    get_invoice, list_invoices, get_invoice_summary, create_invoice, edit_invoice,
    list_customers, list_products,
]]

_quote_tools = [_t(f) for f in [
    list_quotes, create_quote, create_invoice_from_quote, list_customers, list_products,
]]

_customer_tools = [_t(f) for f in [list_customers, create_customer, edit_customer]]

_product_tools = [_t(f) for f in [list_products, create_product, edit_product]]

_email_tools = [_t(f) for f in [send_invoice_by_email, send_quote_by_email]]

_invitation_tools = [_t(f) for f in [invite_user]]

_support_tools = [_t(fetch_support_knowledge)]

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

_SUPPORT_SYSTEM = """You are Billy, an accounting assistant. Answer how-to questions about Billy
by searching the official help docs. Pass 2-3 Danish search terms. Reference source URLs."""

# ---------------------------------------------------------------------------
# Subgraph node functions
# ---------------------------------------------------------------------------


async def invoice_subgraph(state: AgentState) -> AgentState:
    return await run_domain(state, _INVOICE_SYSTEM, _invoice_tools)


async def quote_subgraph(state: AgentState) -> AgentState:
    return await run_domain(state, _QUOTE_SYSTEM, _quote_tools)


async def customer_subgraph(state: AgentState) -> AgentState:
    return await run_domain(state, _CUSTOMER_SYSTEM, _customer_tools)


async def product_subgraph(state: AgentState) -> AgentState:
    return await run_domain(state, _PRODUCT_SYSTEM, _product_tools)


async def email_subgraph(state: AgentState) -> AgentState:
    return await run_domain(state, _EMAIL_SYSTEM, _email_tools)


async def invitation_subgraph(state: AgentState) -> AgentState:
    return await run_domain(state, _INVITATION_SYSTEM, _invitation_tools)


async def support_subgraph(state: AgentState) -> AgentState:
    return await run_domain(state, _SUPPORT_SYSTEM, _support_tools)
