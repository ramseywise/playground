"""Accounting tools — VAT summaries, audit readiness, P&L periods, handoff docs."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import httpx

from app.client import get_client
from app.tools.expenses import _fetch_all_vouchers_for_year
from app.tools.invoices import (
    _fetch_all_invoices_for_year,
    _normalize_date,
    _sevdesk_date,
)


def _in_quarter(date_str: str, month_start: int, month_end: int) -> bool:
    if len(date_str) < 7:
        return False
    try:
        return month_start <= int(date_str[5:7]) <= month_end
    except (ValueError, IndexError):
        return False


async def get_vat_summary(quarter: int, year: int) -> dict:
    """German VAT (Mehrwertsteuer) summary for a reporting quarter.

    Output VAT: VAT collected on open/paid invoices.
    Input VAT: VAT paid on expense vouchers.
    Net VAT payable = output_vat − input_vat.
    Positive = owed to tax authority; negative = refund position.

    Args:
        quarter: 1, 2, 3, or 4.
        year: Calendar year, e.g. 2024.

    Returns:
        Dict with output_vat, input_vat, net_vat_payable, vat_rate_pct,
        currency, quarter, year, and status.
    """
    if quarter not in (1, 2, 3, 4):
        return {"error": "quarter must be 1, 2, 3, or 4"}

    month_start = (quarter - 1) * 3 + 1
    month_end = quarter * 3

    invoices, vouchers = (
        await _fetch_all_invoices_for_year(year),
        await _fetch_all_vouchers_for_year(year),
    )

    output_vat = sum(
        float(inv.get("tax") or 0)
        for inv in invoices
        if inv.get("state") in ("open", "partially_paid", "paid")
        and _in_quarter(inv.get("invoice_date") or "", month_start, month_end)
    )
    input_vat = sum(
        float(v.get("tax") or 0)
        for v in vouchers
        if _in_quarter(v.get("date") or "", month_start, month_end)
    )

    net_vat_payable = round(output_vat - input_vat, 2)

    return {
        "year": year,
        "quarter": quarter,
        "output_vat": round(output_vat, 2),
        "input_vat": round(input_vat, 2),
        "net_vat_payable": net_vat_payable,
        "vat_rate_pct": 19.0,
        "currency": "EUR",
        "status": "payable" if net_vat_payable > 0 else "refund",
    }


async def get_unreconciled_transactions(days_back: int = 30) -> dict:
    """Recent bank transactions from sevdesk, useful for identifying unreconciled items.

    Returns CheckAccountTransactions for the given lookback window. In sevdesk,
    full reconciliation tracking requires ledger bookings — this returns raw
    transaction records for review.

    Args:
        days_back: Days to look back from today (1–365). Default 30.

    Returns:
        Dict with transactions list and total count.
    """
    days_back = max(1, min(days_back, 365))
    cutoff_str = (date.today() - timedelta(days=days_back)).isoformat()

    try:
        resp = await get_client().get(
            "/CheckAccountTransaction",
            params={"startDate": _sevdesk_date(cutoff_str), "limit": 200},
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or []
    transactions = [
        {
            "id": t.get("id"),
            "date": _normalize_date(t.get("valueDate") or t.get("entryDate")),
            "description": t.get("paymtPurpose") or t.get("description"),
            "amount": float(t.get("amount") or 0),
            "type": "credit"
            if (t.get("checkAccountTransactionType") or "").upper() == "R"
            else "debit",
        }
        for t in objects
    ]

    return {
        "days_back": days_back,
        "total_unreconciled": len(transactions),
        "transactions": transactions,
        "currency": "EUR",
    }


async def get_audit_readiness_score() -> dict:
    """Compute an audit readiness score for the sevdesk account (current year).

    Checks performed:
    1. Draft invoices (status 100) — medium severity
    2. Overdue open invoices (due date in the past) — high severity
    3. Vouchers missing description — low severity

    Weighted deductions: high=20, medium=10, low=5 per failing check.
    Score = 100 minus deductions, clamped to 0.

    Returns:
        Dict with score (0–100), checks list, missing_docs, and recommendations.
    """
    today_str = date.today().isoformat()
    invoices = await _fetch_all_invoices_for_year(date.today().year)
    vouchers = await _fetch_all_vouchers_for_year(date.today().year)

    draft_count = sum(1 for inv in invoices if inv.get("state") == "draft")
    overdue_count = sum(
        1
        for inv in invoices
        if inv.get("state") == "open" and (inv.get("due_date") or "") < today_str
    )
    missing_desc = sum(1 for v in vouchers if not (v.get("description") or "").strip())

    checks = [
        {
            "name": "Draft invoices",
            "status": "ok" if draft_count == 0 else "warning",
            "count": draft_count,
            "severity": "medium",
            "description": "Invoices not yet sent to customers",
        },
        {
            "name": "Overdue invoices",
            "status": "ok" if overdue_count == 0 else "critical",
            "count": overdue_count,
            "severity": "high",
            "description": "Open invoices past their due date",
        },
        {
            "name": "Vouchers missing description",
            "status": "ok" if missing_desc == 0 else "info",
            "count": missing_desc,
            "severity": "low",
            "description": "Expense vouchers without description may fail documentation requirements",
        },
    ]

    _deductions = {"high": 20, "medium": 10, "low": 5}
    score = max(
        0,
        100
        - sum(_deductions.get(c["severity"], 5) for c in checks if c["status"] != "ok"),
    )

    missing_docs: list[str] = []
    recommendations: list[str] = []

    if draft_count > 0:
        missing_docs.append(f"{draft_count} draft invoice(s) not yet sent")
        recommendations.append("Send or void pending draft invoices before audit")
    if overdue_count > 0:
        missing_docs.append(f"{overdue_count} overdue invoice(s) awaiting payment")
        recommendations.append("Send payment reminders for overdue invoices")
    if missing_desc > 0:
        missing_docs.append(f"{missing_desc} voucher(s) missing description")
        recommendations.append(
            "Add descriptions to all expense vouchers for documentation"
        )

    if not missing_docs:
        recommendations.append("Account is audit-ready — no action items found.")

    return {
        "score": score,
        "max_score": 100,
        "checks": checks,
        "missing_docs": missing_docs,
        "recommendations": recommendations,
    }


async def get_period_summary(year: int, quarter: Optional[int] = None) -> dict:
    """P&L summary for a year or quarter — suitable for accountant handoff.

    Revenue = open/paid invoice net amounts (excl. VAT).
    Expenses = voucher net amounts (excl. VAT).
    VAT position = output_vat − input_vat (positive = owed to tax authority).

    Args:
        year: Calendar year, e.g. 2024.
        quarter: Optional 1–4 to scope to a quarter. Omit for full year.

    Returns:
        Dict with year, quarter, revenue, expenses, profit, output_vat,
        input_vat, vat_position, invoice_count, expense_count, currency.
    """
    if quarter and quarter not in (1, 2, 3, 4):
        return {"error": "quarter must be 1, 2, 3, or 4"}

    invoices = await _fetch_all_invoices_for_year(year)
    vouchers = await _fetch_all_vouchers_for_year(year)

    if quarter:
        ms, me = (quarter - 1) * 3 + 1, quarter * 3
        invoices = [
            i for i in invoices if _in_quarter(i.get("invoice_date") or "", ms, me)
        ]
        vouchers = [v for v in vouchers if _in_quarter(v.get("date") or "", ms, me)]

    active_invoices = [
        i for i in invoices if i.get("state") in ("open", "partially_paid", "paid")
    ]

    revenue = sum(float(i.get("amount") or 0) for i in active_invoices)
    output_vat = sum(float(i.get("tax") or 0) for i in active_invoices)
    expenses = sum(float(v.get("amount") or 0) for v in vouchers)
    input_vat = sum(float(v.get("tax") or 0) for v in vouchers)

    return {
        "year": year,
        "quarter": quarter,
        "revenue": round(revenue, 2),
        "expenses": round(expenses, 2),
        "profit": round(revenue - expenses, 2),
        "output_vat": round(output_vat, 2),
        "input_vat": round(input_vat, 2),
        "vat_position": round(output_vat - input_vat, 2),
        "invoice_count": len(active_invoices),
        "expense_count": len(vouchers),
        "currency": "EUR",
    }


async def generate_handoff_doc(year: int, quarter: Optional[int] = None) -> dict:
    """Generates a markdown P&L handoff document for accountant review.

    Combines get_period_summary and get_audit_readiness_score into a structured
    markdown document with a P&L summary, VAT table, and documentation status.

    Args:
        year: Calendar year, e.g. 2024.
        quarter: Optional 1–4. Omit for full-year summary.

    Returns:
        Dict with period_label, markdown_summary, and missing_items list.
    """
    period, audit = await _fetch_all_invoices_for_year(year), None  # placeholder
    period = await get_period_summary(year=year, quarter=quarter)
    audit = await get_audit_readiness_score()

    if "error" in period:
        return period

    period_label = f"Q{quarter} {year}" if quarter else str(year)

    lines = [
        f"# Financial Summary — {period_label}",
        "",
        "## Profit & Loss",
        "",
        "| Item | Amount (EUR) |",
        "|------|-------------|",
        f"| Revenue (excl. VAT) | {period['revenue']:,.2f} |",
        f"| Expenses (excl. VAT) | {period['expenses']:,.2f} |",
        f"| **Net Profit** | **{period['profit']:,.2f}** |",
        "",
        "## VAT Summary",
        "",
        "| Item | Amount (EUR) |",
        "|------|-------------|",
        f"| Output VAT (collected) | {period['output_vat']:,.2f} |",
        f"| Input VAT (paid) | {period['input_vat']:,.2f} |",
        f"| **Net VAT Position** | **{period['vat_position']:,.2f}** |",
        "",
        "> Positive = VAT owed to tax authority"
        if period["vat_position"] >= 0
        else "> Negative = VAT refund position",
        "",
        "## Documentation Status",
        "",
        f"Audit score: **{audit['score']}/100**",
        "",
    ]

    if audit["missing_docs"]:
        lines.append("**Outstanding items:**")
        lines.append("")
        for item in audit["missing_docs"]:
            lines.append(f"- {item}")
        lines.append("")
    else:
        lines.append("No outstanding items — ready for review.")
        lines.append("")

    lines += [
        "## Key Figures",
        "",
        f"- Invoices: {period['invoice_count']}",
        f"- Expenses: {period['expense_count']}",
        "",
        "---",
        "_Generated by Clara (sevdesk MCP server)_",
    ]

    return {
        "period_label": period_label,
        "markdown_summary": "\n".join(lines),
        "missing_items": audit["missing_docs"],
    }
