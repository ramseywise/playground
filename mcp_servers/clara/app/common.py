"""Common registration logic — register all Clara tools exactly once."""

from __future__ import annotations

import asyncio
import functools
import logging

from fastmcp import FastMCP

from app.tools.accounting import (
    generate_handoff_doc,
    get_audit_readiness_score,
    get_period_summary,
    get_unreconciled_transactions,
    get_vat_summary,
)
from app.tools.banking import (
    get_bank_balance,
    get_cashflow_forecast,
    get_runway_estimate,
    list_bank_transactions,
    match_transaction_to_invoice,
)
from app.tools.customers import (
    create_customer,
    edit_customer,
    get_customer,
    list_customers,
)
from app.tools.expenses import (
    create_expense,
    get_expense,
    get_expense_summary,
    get_expenses_by_category,
    get_gross_margin,
    get_vendor_spend,
    list_expenses,
)
from app.tools.invoices import (
    create_invoice,
    get_insight_aging_report,
    get_insight_invoice_status,
    get_insight_revenue_summary,
    get_insight_top_customers,
    get_invoice,
    get_invoice_summary,
    list_invoices,
    send_invoice_by_email,
    void_invoice,
)
from app.tools.products import (
    create_product,
    edit_product,
    get_product,
    list_products,
)
from app.tools.insights import (
    detect_anomaly,
    get_break_even_estimate,
    get_customer_concentration,
    get_dso_trend,
    get_margin_by_product,
    get_net_margin,
)
from app.tools.quotes import (
    create_invoice_from_quote,
    create_quote,
    edit_quote,
    get_quote_conversion_stats,
    list_quotes,
)
from app.tools.support_knowledge import fetch_support_knowledge

_log = logging.getLogger("clara.tools")


def _logged(fn):
    """Wrap a tool function with entry/exit logging — async-aware."""
    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def _async(*args, **kwargs):
            _log.info("[MCP] ▶ %s  kwargs=%s", fn.__name__, kwargs)
            try:
                result = await fn(*args, **kwargs)
                _log.info("[MCP] ◀ %s  result=%.300s", fn.__name__, str(result))
                return result
            except Exception:
                _log.exception("[MCP] ✗ %s raised", fn.__name__)
                raise

        return _async

    @functools.wraps(fn)
    def _sync(*args, **kwargs):
        _log.info("[MCP] ▶ %s  kwargs=%s", fn.__name__, kwargs)
        try:
            result = fn(*args, **kwargs)
            _log.info("[MCP] ◀ %s  result=%.300s", fn.__name__, str(result))
            return result
        except Exception:
            _log.exception("[MCP] ✗ %s raised", fn.__name__)
            raise

    return _sync


def register_all(mcp: FastMCP) -> None:
    """Register every Clara tool on *mcp*."""

    # Customers
    mcp.tool()(_logged(list_customers))
    mcp.tool()(_logged(get_customer))
    mcp.tool()(_logged(create_customer))  # [WRITE]
    mcp.tool()(_logged(edit_customer))  # [WRITE]

    # Invoices
    mcp.tool()(_logged(list_invoices))
    mcp.tool()(_logged(get_invoice))
    mcp.tool()(_logged(get_invoice_summary))
    mcp.tool()(_logged(create_invoice))  # [WRITE]
    mcp.tool()(_logged(void_invoice))  # [WRITE]
    mcp.tool()(_logged(send_invoice_by_email))  # [WRITE] sends real email

    # Invoice insights
    mcp.tool()(_logged(get_insight_revenue_summary))
    mcp.tool()(_logged(get_insight_invoice_status))
    mcp.tool()(_logged(get_insight_top_customers))
    mcp.tool()(_logged(get_insight_aging_report))

    # Products
    mcp.tool()(_logged(list_products))
    mcp.tool()(_logged(get_product))
    mcp.tool()(_logged(create_product))  # [WRITE]
    mcp.tool()(_logged(edit_product))  # [WRITE]

    # Quotes
    mcp.tool()(_logged(list_quotes))
    mcp.tool()(_logged(create_quote))  # [WRITE]
    mcp.tool()(_logged(edit_quote))  # [WRITE]
    mcp.tool()(_logged(create_invoice_from_quote))  # [WRITE]
    mcp.tool()(_logged(get_quote_conversion_stats))

    # Expenses
    mcp.tool()(_logged(list_expenses))
    mcp.tool()(_logged(get_expense))
    mcp.tool()(_logged(create_expense))  # [WRITE]
    mcp.tool()(_logged(get_expense_summary))
    mcp.tool()(_logged(get_vendor_spend))
    mcp.tool()(_logged(get_expenses_by_category))
    mcp.tool()(_logged(get_gross_margin))

    # Banking
    mcp.tool()(_logged(get_bank_balance))
    mcp.tool()(_logged(list_bank_transactions))
    mcp.tool()(_logged(match_transaction_to_invoice))  # [WRITE]
    mcp.tool()(_logged(get_cashflow_forecast))
    mcp.tool()(_logged(get_runway_estimate))

    # Cross-domain insights
    mcp.tool()(_logged(get_net_margin))
    mcp.tool()(_logged(get_margin_by_product))
    mcp.tool()(_logged(get_customer_concentration))
    mcp.tool()(_logged(get_dso_trend))
    mcp.tool()(_logged(get_break_even_estimate))
    mcp.tool()(_logged(detect_anomaly))

    # Support knowledge
    mcp.tool()(_logged(fetch_support_knowledge))

    # Accounting
    mcp.tool()(_logged(get_vat_summary))
    mcp.tool()(_logged(get_unreconciled_transactions))
    mcp.tool()(_logged(get_audit_readiness_score))
    mcp.tool()(_logged(get_period_summary))
    mcp.tool()(_logged(generate_handoff_doc))
