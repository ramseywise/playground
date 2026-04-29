"""End-to-end connectivity tests via FastMCP in-memory Client."""

from app.main_noauth import mcp
from fastmcp import Client

# All tools currently registered in common.register_all().
# Tools marked [WRITE] require a live Billy account — they work against the
# local SQLite stub but are disabled in the VA agents until a test account
# is available. TODO(2): audit this list when write tools are re-enabled.
ALL_TOOLS = {
    # Customers
    "list_customers",
    "get_customer",
    "create_customer",
    "edit_customer",  # [WRITE]
    # Invoices
    "get_invoice",
    "list_invoices",
    "get_invoice_summary",
    "get_invoice_dso_stats",
    "get_invoice_lines_summary",
    "create_invoice",
    "edit_invoice",
    "void_invoice",  # [WRITE]
    "send_invoice_reminder",  # [WRITE]
    # Invoice insights
    "get_insight_revenue_summary",
    "get_insight_invoice_status",
    "get_insight_monthly_revenue",
    "get_insight_top_customers",
    "get_insight_aging_report",
    "get_insight_customer_summary",
    "get_insight_product_revenue",
    # Products
    "list_products",
    "get_product",
    "create_product",
    "edit_product",  # [WRITE]
    # Quotes
    "list_quotes",
    "get_quote_conversion_stats",
    "create_quote",
    "edit_quote",
    "create_invoice_from_quote",  # [WRITE]
    # Emails
    "send_invoice_by_email",
    "send_quote_by_email",  # [WRITE]
    # Invitations
    "invite_user",  # [WRITE]
    # Expenses
    "list_expenses",
    "get_expense",
    "get_expense_summary",
    "get_vendor_spend",
    "get_expenses_by_category",
    "get_gross_margin",
    "create_expense",  # [WRITE]
    # Banking
    "get_bank_balance",
    "list_bank_transactions",
    "get_cashflow_forecast",
    "get_runway_estimate",
    "match_transaction_to_invoice",  # [WRITE]
    # Cross-domain insights
    "get_net_margin",
    "get_margin_by_product",
    "get_customer_concentration",
    "get_dso_trend",
    "get_break_even_estimate",
    "detect_anomaly",
    # Accounting
    "get_vat_summary",
    "get_unreconciled_transactions",
    "get_audit_readiness_score",
    "get_period_summary",
    "generate_handoff_doc",
    # Support
    "fetch_support_knowledge",
}


class TestServerConnectivity:
    async def test_ping(self):
        async with Client(mcp) as c:
            assert await c.ping() is True

    async def test_all_tools_registered(self):
        async with Client(mcp) as c:
            tools = await c.list_tools()
            names = {t.name for t in tools}
            assert ALL_TOOLS == names
