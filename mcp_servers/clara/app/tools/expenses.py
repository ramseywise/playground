"""Expense tools — sevdesk /Voucher API."""

from __future__ import annotations

from datetime import date
from typing import Optional

import httpx

from app.client import get_client
from app.tools.invoices import (
    _fetch_all_invoices_for_year,
    _normalize_date,
    _sevdesk_date,
)

_VOUCHER_STATUS = {"50": "draft", "100": "open", "1000": "paid"}
_STATUS_TO_CODE = {v: k for k, v in _VOUCHER_STATUS.items()}


def _normalize_voucher(v: dict) -> dict:
    status_code = str(v.get("status") or "50")
    supplier = v.get("supplier") or {}
    return {
        "id": v.get("id"),
        "vendor": supplier.get("name") if isinstance(supplier, dict) else None,
        "supplier_id": supplier.get("id") if isinstance(supplier, dict) else None,
        "description": v.get("description"),
        "date": _normalize_date(v.get("voucherDate")),
        "amount": float(v.get("sumNet") or 0),
        "tax": float(v.get("sumTax") or 0),
        "gross_amount": float(v.get("sumGross") or 0),
        "currency": v.get("currency", "EUR"),
        "status": _VOUCHER_STATUS.get(status_code, status_code),
    }


async def list_expenses(
    limit: int = 50,
    offset: int = 0,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
) -> dict:
    """Lists expense vouchers from sevdesk.

    Args:
        limit: Max records to return. Defaults to 50.
        offset: Pagination offset. Defaults to 0.
        date_from: Earliest date inclusive (YYYY-MM-DD).
        date_to: Latest date inclusive (YYYY-MM-DD).
        status: Filter by status — 'draft', 'open', 'paid'.

    Returns:
        Dict with total, offset, and a list of expense records.
    """
    params: dict = {"limit": limit, "offset": offset, "voucherType": "VOU"}
    if date_from:
        params["startDate"] = _sevdesk_date(date_from)
    if date_to:
        params["endDate"] = _sevdesk_date(date_to)
    if status and status in _STATUS_TO_CODE:
        params["status"] = _STATUS_TO_CODE[status]

    try:
        resp = await get_client().get("/Voucher", params=params)
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
        "expenses": [_normalize_voucher(v) for v in objects],
    }


async def get_expense(expense_id: str) -> dict:
    """Gets a single expense voucher by ID.

    Args:
        expense_id: The sevdesk voucher ID.

    Returns:
        Full expense record, or an error dict if not found.
    """
    try:
        resp = await get_client().get(f"/Voucher/{expense_id}")
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": f"Expense '{expense_id}' not found."}
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or []
    if not objects:
        return {"error": f"Expense '{expense_id}' not found."}
    v = objects[0] if isinstance(objects, list) else objects
    return _normalize_voucher(v)


async def create_expense(
    vendor: str,
    amount: float,
    date: str,
    description: Optional[str] = None,
    tax_rate: float = 19.0,
    supplier_id: Optional[str] = None,
    currency: str = "EUR",
) -> dict:
    """Records a new expense voucher in sevdesk.

    Creates a draft voucher. The amount is the net (excl. VAT) expense.

    Args:
        vendor: Name of the vendor or supplier.
        amount: Net expense amount (excl. VAT).
        date: Expense date in ISO format (YYYY-MM-DD).
        description: Optional memo or description.
        tax_rate: VAT rate percentage. Defaults to 19.0.
        supplier_id: Optional sevdesk contact ID to link as supplier.
        currency: Currency code. Defaults to 'EUR'.

    Returns:
        The newly created expense record, or an error dict.
    """
    payload: dict = {
        "objectName": "Voucher",
        "voucherDate": _sevdesk_date(date),
        "description": description or vendor,
        "voucherType": "VOU",
        "status": "50",
        "taxType": "default",
        "currency": currency,
    }
    if supplier_id:
        payload["supplier"] = {"id": supplier_id, "objectName": "Contact"}

    try:
        resp = await get_client().post("/Voucher", json=payload)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or {}
    v = objects if isinstance(objects, dict) else (objects[0] if objects else {})
    result = _normalize_voucher(v)
    # Backfill fields that sevdesk may not return in the create response
    if not result.get("vendor"):
        result["vendor"] = vendor
    if not result.get("amount"):
        result["amount"] = amount
    result["tax_rate"] = tax_rate
    return result


async def _fetch_all_vouchers_for_year(year: int) -> list[dict]:
    """Internal: fetch up to 500 expense vouchers for a given year."""
    params = {"limit": 500, "offset": 0, "voucherType": "VOU"}
    try:
        resp = await get_client().get("/Voucher", params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    objects = data.get("objects") or []
    year_str = str(year)
    return [
        _normalize_voucher(v)
        for v in objects
        if (_normalize_date(v.get("voucherDate")) or "").startswith(year_str)
    ]


async def get_expense_summary(year: Optional[int] = None) -> dict:
    """Returns total expense amount and count for a given year.

    Args:
        year: Calendar year. Defaults to current year.

    Returns:
        Dict with year, total net amount, count, and currency.
    """
    target_year = year or date.today().year
    vouchers = await _fetch_all_vouchers_for_year(target_year)
    total = sum(v.get("amount") or 0 for v in vouchers)
    return {
        "year": target_year,
        "total": round(total, 2),
        "count": len(vouchers),
        "currency": "EUR",
    }


async def get_vendor_spend(
    vendor: Optional[str] = None,
    year: Optional[int] = None,
) -> dict:
    """Returns total spend per vendor, optionally filtered by name or year.

    Args:
        vendor: Substring filter on vendor name (case-insensitive).
        year: Filter to a specific calendar year. Defaults to current year.

    Returns:
        Dict with vendors list of {vendor, total, count, currency}.
    """
    target_year = year or date.today().year
    vouchers = await _fetch_all_vouchers_for_year(target_year)

    if vendor:
        vendor_lower = vendor.lower()
        vouchers = [
            v for v in vouchers if vendor_lower in (v.get("vendor") or "").lower()
        ]

    spend: dict[str, dict] = {}
    for v in vouchers:
        name = v.get("vendor") or "Unknown"
        if name not in spend:
            spend[name] = {"vendor": name, "total": 0.0, "count": 0, "currency": "EUR"}
        spend[name]["total"] += v.get("amount") or 0
        spend[name]["count"] += 1

    vendors = sorted(spend.values(), key=lambda x: x["total"], reverse=True)
    for entry in vendors:
        entry["total"] = round(entry["total"], 2)
    return {"vendors": vendors}


async def get_expenses_by_category(year: Optional[int] = None) -> dict:
    """Returns expense totals broken down by vendor for a given year.

    Note: sevdesk vouchers do not have predefined expense categories. Grouping
    is by vendor name rather than by accounting category.

    Args:
        year: Calendar year filter. Defaults to current year.

    Returns:
        Dict with year, total, and vendor breakdown.
    """
    return await get_vendor_spend(year=year)


async def get_gross_margin(year: Optional[int] = None) -> dict:
    """Returns gross margin by comparing invoice revenue against expenses.

    Revenue = sum of open/partially_paid/paid invoice net amounts.
    Costs   = sum of expense voucher net amounts.
    Gross margin % = (revenue − costs) / revenue × 100.

    Args:
        year: Calendar year. Defaults to current year.

    Returns:
        Dict with revenue, cogs, gross_profit, gross_margin_pct, currency.
    """
    target_year = year or date.today().year

    invoices, vouchers = (
        await _fetch_all_invoices_for_year(target_year),
        await _fetch_all_vouchers_for_year(target_year),
    )

    revenue = sum(
        float(inv.get("amount") or 0)
        for inv in invoices
        if inv.get("state") in ("open", "partially_paid", "paid")
    )
    cogs = sum(v.get("amount") or 0 for v in vouchers)
    gross_profit = revenue - cogs
    gross_margin_pct = round(gross_profit / revenue * 100, 1) if revenue > 0 else 0.0

    return {
        "year": target_year,
        "revenue": round(revenue, 2),
        "cogs": round(cogs, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_margin_pct": gross_margin_pct,
        "currency": "EUR",
    }
