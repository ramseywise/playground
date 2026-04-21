"""Banking stub tools for the Billy MCP server."""

from __future__ import annotations

from datetime import date
from typing import Optional

from app.db import get_conn, next_id


def get_bank_balance() -> dict:
    """Returns current balances for all bank accounts.

    Returns:
        Dict with accounts list of {account_id, name, bank_name, balance,
        account_no, currency} and total_balance.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, bank_name, balance, account_no, currency FROM bank_accounts"
        ).fetchall()
        total = conn.execute(
            "SELECT COALESCE(SUM(balance), 0) FROM bank_accounts WHERE currency = 'DKK'"
        ).fetchone()[0]

    accounts = [dict(r) for r in rows]
    return {"accounts": accounts, "total_balance": total, "currency": "DKK"}


def list_bank_transactions(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    account_id: Optional[str] = None,
    transaction_type: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> dict:
    """Lists bank transactions with optional filters.

    Returns a paginated list of transactions sorted by date descending.

    Args:
        date_from: Earliest date inclusive (ISO format, e.g. '2024-01-01').
        date_to: Latest date inclusive (ISO format, e.g. '2024-12-31').
        account_id: Filter to a specific bank account ID.
        transaction_type: Filter by type — 'credit' (money in) or 'debit' (money out).
        page: Page number (1-based). Defaults to 1.
        page_size: Items per page. Defaults to 50.

    Returns:
        Dict with total, page, page_count, and list of transaction records.
    """
    conditions: list[str] = []
    params: list = []

    if date_from:
        conditions.append("date >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("date <= ?")
        params.append(date_to)
    if account_id:
        conditions.append("account_id = ?")
        params.append(account_id)
    if transaction_type in ("credit", "debit"):
        conditions.append("type = ?")
        params.append(transaction_type)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_conn() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM bank_transactions {where}", params
        ).fetchone()[0]
        offset = (page - 1) * page_size
        rows = conn.execute(
            f"SELECT * FROM bank_transactions {where} ORDER BY date DESC, id DESC LIMIT ? OFFSET ?",
            params + [page_size, offset],
        ).fetchall()

    return {
        "total": total,
        "page": page,
        "page_count": max(1, (total + page_size - 1) // page_size),
        "transactions": [dict(r) for r in rows],
    }


def match_transaction_to_invoice(transaction_id: str, invoice_id: str) -> dict:
    """Links a bank transaction to an invoice to mark it as reconciled.

    Args:
        transaction_id: The ID of the bank transaction to reconcile.
        invoice_id: The ID of the invoice this payment belongs to.

    Returns:
        Dict with matched status, transaction_id, and invoice_id. Returns an
        error dict if either record is not found.
    """
    with get_conn() as conn:
        txn = conn.execute(
            "SELECT id FROM bank_transactions WHERE id = ?", (transaction_id,)
        ).fetchone()
        if not txn:
            return {"error": f"Transaction '{transaction_id}' not found."}
        inv = conn.execute(
            "SELECT id FROM invoices WHERE id = ?", (invoice_id,)
        ).fetchone()
        if not inv:
            return {"error": f"Invoice '{invoice_id}' not found."}
        conn.execute(
            "UPDATE bank_transactions SET matched_invoice_id = ? WHERE id = ?",
            (invoice_id, transaction_id),
        )
    return {"matched": True, "transaction_id": transaction_id, "invoice_id": invoice_id}


def get_cashflow_forecast(months: int = 3) -> dict:
    """Projects cash inflow and outflow for each of the next N months.

    Inflow = approved unpaid invoices with due dates in each future month (gross amount).
    Outflow = average monthly expenses based on all historical expense data.

    Args:
        months: Number of future months to forecast. Defaults to 3.

    Returns:
        Dict with currency and forecast list of {month, projected_inflow,
        projected_outflow, net}.
    """
    months = max(1, min(months, 12))
    today = date.today()

    with get_conn() as conn:
        avg_row = conn.execute(
            """SELECT COALESCE(AVG(monthly_total), 0)
               FROM (
                   SELECT strftime('%Y-%m', date) AS month, SUM(amount) AS monthly_total
                   FROM expenses
                   GROUP BY month
               )"""
        ).fetchone()
        avg_monthly_burn = round(avg_row[0], 2)

        future_rows = conn.execute(
            """SELECT strftime('%Y-%m', due_date) AS month, SUM(gross_amount) AS inflow
               FROM invoices
               WHERE state = 'approved' AND is_paid = 0 AND due_date >= ?
               GROUP BY month""",
            (today.isoformat(),),
        ).fetchall()

    future_inflows = {r[0]: r[1] for r in future_rows}

    forecast = []
    for i in range(1, months + 1):
        raw = today.month - 1 + i
        year = today.year + raw // 12
        month = raw % 12 + 1
        label = f"{year}-{month:02d}"
        inflow = round(future_inflows.get(label, 0.0), 2)
        net = round(inflow - avg_monthly_burn, 2)
        forecast.append(
            {
                "month": label,
                "projected_inflow": inflow,
                "projected_outflow": avg_monthly_burn,
                "net": net,
            }
        )

    return {"currency": "DKK", "forecast": forecast}


def get_runway_estimate() -> dict:
    """Estimates operating runway based on current bank balance and average monthly burn.

    Runway months = total bank balance ÷ average monthly expenses.
    Average monthly burn is derived from all historical expense records.

    Returns:
        Dict with balance, avg_monthly_burn, runway_months (None if no expense
        history), and currency.
    """
    with get_conn() as conn:
        balance = conn.execute(
            "SELECT COALESCE(SUM(balance), 0) FROM bank_accounts"
        ).fetchone()[0]

        avg_row = conn.execute(
            """SELECT COALESCE(AVG(monthly_total), 0)
               FROM (
                   SELECT strftime('%Y-%m', date) AS month, SUM(amount) AS monthly_total
                   FROM expenses
                   GROUP BY month
               )"""
        ).fetchone()
        avg_monthly_burn = round(avg_row[0], 2)

    runway_months = (
        round(balance / avg_monthly_burn, 1) if avg_monthly_burn > 0 else None
    )

    return {
        "balance": balance,
        "avg_monthly_burn": avg_monthly_burn,
        "runway_months": runway_months,
        "currency": "DKK",
    }
