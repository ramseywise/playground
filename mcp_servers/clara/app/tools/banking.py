"""Banking tools — sevdesk /CheckAccount and /CheckAccountTransaction APIs."""

from __future__ import annotations

from datetime import date
from typing import Optional

import httpx

from app.client import get_client
from app.tools.expenses import _fetch_all_vouchers_for_year
from app.tools.invoices import _normalize_date, _sevdesk_date, list_invoices


def _normalize_account(a: dict) -> dict:
    return {
        "account_id": a.get("id"),
        "name": a.get("name"),
        "bank_name": a.get("bankName"),
        "balance": float(a.get("balance") or 0),
        "account_no": a.get("accountNumber"),
        "currency": a.get("currency", "EUR"),
    }


def _normalize_transaction(t: dict) -> dict:
    raw_type = (t.get("checkAccountTransactionType") or "").upper()
    tx_type = (
        "credit" if raw_type == "R" else ("debit" if raw_type == "Z" else raw_type)
    )
    account = t.get("checkAccount") or {}
    return {
        "id": t.get("id"),
        "date": _normalize_date(t.get("valueDate") or t.get("entryDate")),
        "description": t.get("paymtPurpose") or t.get("description"),
        "amount": float(t.get("amount") or 0),
        "type": tx_type,
        "account_id": account.get("id") if isinstance(account, dict) else None,
    }


async def get_bank_balance() -> dict:
    """Returns current balances for all bank accounts in sevdesk.

    Returns:
        Dict with accounts list of {account_id, name, bank_name, balance,
        account_no, currency} and total_balance.
    """
    try:
        resp = await get_client().get("/CheckAccount", params={"limit": 100})
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or []
    accounts = [_normalize_account(a) for a in objects]
    total = round(sum(a["balance"] for a in accounts), 2)
    return {"accounts": accounts, "total_balance": total, "currency": "EUR"}


async def list_bank_transactions(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    account_id: Optional[str] = None,
    transaction_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Lists bank transactions with optional filters.

    Args:
        date_from: Earliest date inclusive (YYYY-MM-DD).
        date_to: Latest date inclusive (YYYY-MM-DD).
        account_id: Filter to a specific bank account ID.
        transaction_type: Filter by type — 'credit' (money in) or 'debit' (money out).
        limit: Max records to return. Defaults to 50.
        offset: Pagination offset. Defaults to 0.

    Returns:
        Dict with total, offset, and a list of transaction records.
    """
    params: dict = {"limit": limit, "offset": offset}
    if date_from:
        params["startDate"] = _sevdesk_date(date_from)
    if date_to:
        params["endDate"] = _sevdesk_date(date_to)
    if account_id:
        params["checkAccount[id]"] = account_id
        params["checkAccount[objectName]"] = "CheckAccount"
    if transaction_type == "credit":
        params["checkAccountTransactionType"] = "R"
    elif transaction_type == "debit":
        params["checkAccountTransactionType"] = "Z"

    try:
        resp = await get_client().get("/CheckAccountTransaction", params=params)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or []
    return {
        "total": data.get("total", len(objects)),
        "offset": offset,
        "transactions": [_normalize_transaction(t) for t in objects],
    }


async def match_transaction_to_invoice(transaction_id: str, invoice_id: str) -> dict:
    """Links a bank transaction to an invoice by marking the invoice as paid.

    Note: sevdesk does not have a direct transaction-to-invoice link endpoint.
    This tool marks the invoice status as paid (1000) as a reconciliation proxy.

    Args:
        transaction_id: The sevdesk CheckAccountTransaction ID.
        invoice_id: The sevdesk invoice ID to mark as paid.

    Returns:
        Dict with matched=True, transaction_id, and invoice_id.
    """
    try:
        resp = await get_client().put(
            f"/Invoice/{invoice_id}",
            json={"objectName": "Invoice", "status": "1000"},
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": f"Invoice '{invoice_id}' not found."}
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    return {
        "matched": True,
        "transaction_id": transaction_id,
        "invoice_id": invoice_id,
        "note": "Invoice marked paid. Full ledger reconciliation requires a booking entry in sevdesk.",
    }


async def _avg_monthly_voucher_spend() -> float:
    """Internal: average monthly net spend from the last two years of vouchers."""
    today = date.today()
    all_vouchers: list[dict] = []
    for yr in [today.year, today.year - 1]:
        all_vouchers.extend(await _fetch_all_vouchers_for_year(yr))

    monthly: dict[str, float] = {}
    for v in all_vouchers:
        dt = v.get("date") or ""
        if len(dt) >= 7:
            month = dt[:7]
            monthly[month] = monthly.get(month, 0.0) + float(v.get("amount") or 0)

    if not monthly:
        return 0.0
    return round(sum(monthly.values()) / len(monthly), 2)


async def get_cashflow_forecast(months: int = 3) -> dict:
    """Projects cash inflow and outflow for each of the next N months.

    Inflow = open (unpaid) invoices with due dates in each future month.
    Outflow = average monthly expense spend from historical voucher data.

    Args:
        months: Number of future months to forecast (1–12). Defaults to 3.

    Returns:
        Dict with currency and forecast list of {month, projected_inflow,
        projected_outflow, net}.
    """
    months = max(1, min(months, 12))
    today = date.today()
    today_str = today.isoformat()

    result = await list_invoices(limit=500, offset=0, state="open")
    open_invoices = result.get("invoices", [])

    future_inflows: dict[str, float] = {}
    for inv in open_invoices:
        due = inv.get("due_date") or ""
        if due >= today_str and len(due) >= 7:
            month = due[:7]
            future_inflows[month] = future_inflows.get(month, 0.0) + float(
                inv.get("gross_amount") or 0
            )

    avg_outflow = await _avg_monthly_voucher_spend()

    forecast = []
    for i in range(1, months + 1):
        raw = today.month - 1 + i
        yr = today.year + raw // 12
        mo = raw % 12 + 1
        label = f"{yr}-{mo:02d}"
        inflow = round(future_inflows.get(label, 0.0), 2)
        forecast.append(
            {
                "month": label,
                "projected_inflow": inflow,
                "projected_outflow": avg_outflow,
                "net": round(inflow - avg_outflow, 2),
            }
        )

    return {"currency": "EUR", "forecast": forecast}


async def get_runway_estimate() -> dict:
    """Estimates operating runway based on current bank balance and average monthly burn.

    Runway months = total bank balance / average monthly expense spend.

    Returns:
        Dict with balance, avg_monthly_burn, runway_months (None if no expense
        history), and currency.
    """
    balance_data = await get_bank_balance()
    if "error" in balance_data:
        return balance_data

    balance = balance_data.get("total_balance", 0.0)
    avg_monthly_burn = await _avg_monthly_voucher_spend()
    runway_months = (
        round(balance / avg_monthly_burn, 1) if avg_monthly_burn > 0 else None
    )

    return {
        "balance": balance,
        "avg_monthly_burn": avg_monthly_burn,
        "runway_months": runway_months,
        "currency": "EUR",
    }
