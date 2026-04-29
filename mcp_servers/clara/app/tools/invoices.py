"""Invoice tools — sevdesk /Invoice API."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import httpx
from pydantic import BaseModel

from app.client import get_client

# sevdesk status codes → normalized state names (mirrors Billy's states)
_STATUS_MAP = {"100": "draft", "200": "open", "750": "partially_paid", "1000": "paid"}
_STATE_TO_STATUS = {v: k for k, v in _STATUS_MAP.items()}

# Default tax rate for Germany (19 %)
_DEFAULT_TAX_RATE = 19.0
# sevdesk unity ID 1 = "Stück" (piece/unit)
_UNITY_ID = "1"


class InvoiceLine(BaseModel):
    name: str
    quantity: float = 1.0
    unit_price: float
    tax_rate: float = _DEFAULT_TAX_RATE


def _sevdesk_date(iso: str) -> str:
    """YYYY-MM-DD → DD.MM.YYYY for sevdesk create/update payloads."""
    d = date.fromisoformat(iso)
    return d.strftime("%d.%m.%Y")


def _normalize_date(raw: str | None) -> str:
    """Normalise sevdesk date strings (ISO timestamp or DD.MM.YYYY) → YYYY-MM-DD."""
    if not raw:
        return ""
    if "T" in raw:
        return raw[:10]
    if "." in raw:
        parts = raw.split(".")
        try:
            return f"{parts[2]}-{parts[1]}-{parts[0]}"
        except IndexError:
            return raw
    return raw


def _normalize_invoice(inv: dict) -> dict:
    status_code = str(inv.get("status", ""))
    contact = inv.get("contact") or {}
    return {
        "id": inv.get("id"),
        "invoice_no": inv.get("invoiceNumber"),
        "contact_id": contact.get("id") if isinstance(contact, dict) else None,
        "customer_name": inv.get("contactName"),
        "invoice_date": _normalize_date(inv.get("invoiceDate")),
        "due_date": _normalize_date(inv.get("dueDate")),
        "state": _STATUS_MAP.get(status_code, status_code),
        "amount": float(inv.get("sumNet") or 0),
        "tax": float(inv.get("sumTax") or 0),
        "gross_amount": float(inv.get("sumGross") or 0),
        "currency": inv.get("currency", "EUR"),
    }


async def list_invoices(
    limit: int = 50,
    offset: int = 0,
    state: Optional[str] = None,
    contact_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Lists invoices from sevdesk.

    Args:
        limit: Max records to return. Defaults to 50.
        offset: Pagination offset. Defaults to 0.
        state: Filter by state — 'draft', 'open', 'partially_paid', or 'paid'.
        contact_id: Filter by sevdesk contact ID.
        start_date: Earliest invoice date to include (YYYY-MM-DD).
        end_date: Latest invoice date to include (YYYY-MM-DD).

    Returns:
        Dict with total, offset, and a list of invoice records.
    """
    params: dict = {"limit": limit, "offset": offset}
    if state and state in _STATE_TO_STATUS:
        params["status"] = _STATE_TO_STATUS[state]
    if contact_id:
        params["contact[id]"] = contact_id
        params["contact[objectName]"] = "Contact"
    if start_date:
        params["invoiceDateFrom"] = _sevdesk_date(start_date)
    if end_date:
        params["invoiceDateTo"] = _sevdesk_date(end_date)

    try:
        resp = await get_client().get("/Invoice", params=params)
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
        "invoices": [_normalize_invoice(inv) for inv in objects],
    }


async def get_invoice(invoice_id: str) -> dict:
    """Gets detailed information about a single invoice by ID.

    Args:
        invoice_id: The sevdesk invoice ID.

    Returns:
        Full invoice record with positions, or an error dict if not found.
    """
    try:
        resp = await get_client().get(f"/Invoice/{invoice_id}")
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": f"Invoice '{invoice_id}' not found."}
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or []
    if not objects:
        return {"error": f"Invoice '{invoice_id}' not found."}

    inv = objects[0] if isinstance(objects, list) else objects
    normalized = _normalize_invoice(inv)

    # Fetch positions (line items)
    try:
        pos_resp = await get_client().get(
            "/InvoicePos",
            params={"invoice[id]": invoice_id, "invoice[objectName]": "Invoice"},
        )
        pos_resp.raise_for_status()
        pos_data = pos_resp.json()
        positions = pos_data.get("objects") or []
        normalized["lines"] = [
            {
                "id": p.get("id"),
                "name": p.get("name"),
                "quantity": float(p.get("quantity") or 1),
                "unit_price": float(p.get("price") or 0),
                "tax_rate": float(p.get("taxRate") or 0),
                "amount": float(p.get("sumNet") or 0),
            }
            for p in positions
        ]
    except (httpx.HTTPStatusError, httpx.RequestError):
        normalized["lines"] = []

    return normalized


async def create_invoice(
    contact_id: str,
    lines: list[InvoiceLine],
    entry_date: Optional[str] = None,
    payment_terms_days: int = 14,
    currency: str = "EUR",
    state: str = "open",
) -> dict:
    """Creates a new invoice in sevdesk via the Factory endpoint.

    Args:
        contact_id: The sevdesk contact ID to bill.
        lines: Invoice line items with name, quantity, unit_price, and tax_rate.
        entry_date: Invoice date (YYYY-MM-DD). Defaults to today.
        payment_terms_days: Days until due. Defaults to 14.
        currency: Currency code. Defaults to 'EUR'.
        state: Initial state — 'draft' or 'open'. Defaults to 'open'.

    Returns:
        The newly created invoice record, or an error dict.
    """
    inv_date = entry_date or date.today().isoformat()
    due_date = (
        date.fromisoformat(inv_date) + timedelta(days=payment_terms_days)
    ).isoformat()
    status_code = _STATE_TO_STATUS.get(state, "200")

    payload = {
        "invoice": {
            "objectName": "Invoice",
            "mapAll": True,
            "contact": {"id": contact_id, "objectName": "Contact"},
            "invoiceDate": _sevdesk_date(inv_date),
            "dueDate": _sevdesk_date(due_date),
            "status": status_code,
            "currency": currency,
            "invoiceType": "RE",
        },
        "invoicePosSave": [
            {
                "objectName": "InvoicePos",
                "mapAll": True,
                "name": ln.name,
                "quantity": ln.quantity,
                "price": ln.unit_price,
                "taxRate": ln.tax_rate,
                "unity": {"id": _UNITY_ID, "objectName": "Unity"},
            }
            for ln in lines
        ],
        "invoicePosDelete": None,
        "filename": None,
        "discountSave": None,
        "discountDelete": None,
    }

    try:
        resp = await get_client().post("/Invoice/Factory/saveInvoice", json=payload)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    inv = (data.get("objects") or {}).get("invoice") or {}
    return _normalize_invoice(inv)


async def void_invoice(invoice_id: str) -> dict:
    """Cancels (voids) an invoice in sevdesk.

    Args:
        invoice_id: The sevdesk invoice ID to cancel.

    Returns:
        Dict with voided=True and invoice_id, or an error dict.
    """
    try:
        resp = await get_client().put(f"/Invoice/{invoice_id}/cancelInvoice")
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": f"Invoice '{invoice_id}' not found."}
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    return {"voided": True, "invoice_id": invoice_id}


async def send_invoice_by_email(
    invoice_id: str,
    to_email: str,
    subject: Optional[str] = None,
    body: Optional[str] = None,
) -> dict:
    """Sends an invoice to a customer by email via sevdesk.

    Args:
        invoice_id: The sevdesk invoice ID to send.
        to_email: Recipient email address.
        subject: Optional email subject (sevdesk default used if omitted).
        body: Optional email body text.

    Returns:
        Dict with sent=True, or an error dict.
    """
    payload: dict = {
        "sendType": "VM",  # VM = email
        "toEmail": to_email,
    }
    if subject:
        payload["subject"] = subject
    if body:
        payload["text"] = body

    try:
        resp = await get_client().post(f"/Invoice/{invoice_id}/sendBy", json=payload)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    return {"sent": True, "invoice_id": invoice_id, "to": to_email}


# ---------------------------------------------------------------------------
# Insight helpers — computed from live API data
# ---------------------------------------------------------------------------


async def _fetch_all_invoices_for_year(fiscal_year: int) -> list[dict]:
    """Fetches all invoices for the given year (paginates up to 500)."""
    start = f"{fiscal_year}-01-01"
    end = f"{fiscal_year}-12-31"
    result = await list_invoices(limit=500, offset=0, start_date=start, end_date=end)
    return result.get("invoices", [])


async def get_invoice_summary(
    fiscal_year: Optional[int] = None,
    month: Optional[int] = None,
) -> dict:
    """Aggregate invoice counts and amounts by state for the fiscal year.

    Args:
        fiscal_year: Year to summarise. Defaults to current year.
        month: Optional month (1–12) to narrow the period.

    Returns:
        Dict with counts and amounts for all, draft, open, paid, and overdue invoices.
    """
    year = fiscal_year or date.today().year
    invoices = await _fetch_all_invoices_for_year(year)
    today = date.today().isoformat()

    if month is not None:
        prefix = f"{year}-{month:02d}"
        invoices = [
            inv for inv in invoices if inv.get("invoice_date", "").startswith(prefix)
        ]

    def _agg(subset: list[dict]) -> dict:
        return {
            "count": len(subset),
            "amount": round(sum(i["gross_amount"] for i in subset), 2),
        }

    all_inv = invoices
    draft = [i for i in invoices if i["state"] == "draft"]
    open_ = [i for i in invoices if i["state"] == "open"]
    paid = [i for i in invoices if i["state"] == "paid"]
    overdue = [
        i for i in invoices if i["state"] == "open" and i.get("due_date", "") < today
    ]

    result: dict = {
        "fiscal_year": year,
        "all": _agg(all_inv),
        "draft": _agg(draft),
        "open": _agg(open_),
        "paid": _agg(paid),
        "overdue": _agg(overdue),
    }
    if month is not None:
        result["month"] = month
    return result


async def get_insight_revenue_summary(
    fiscal_year: Optional[int] = None,
    month: Optional[int] = None,
) -> dict:
    """Revenue KPI cards: total invoiced, collected, outstanding, overdue.

    Args:
        fiscal_year: Year to summarise. Defaults to current year.
        month: Optional month (1–12) to narrow the period.

    Returns:
        Dict with KPI cards matching Billy's insight format.
    """
    year = fiscal_year or date.today().year
    cur = await get_invoice_summary(year, month)

    result: dict = {
        "fiscalYear": year,
        "currency": "EUR",
        "cards": [
            {"label": "Total invoiced", "amount": cur["all"]["amount"], "delta": None},
            {"label": "Collected", "amount": cur["paid"]["amount"], "delta": None},
            {"label": "Outstanding", "amount": cur["open"]["amount"], "delta": None},
            {"label": "Overdue", "amount": cur["overdue"]["amount"], "delta": None},
        ],
    }
    if month is not None:
        result["month"] = month
    return result


async def get_insight_invoice_status(fiscal_year: Optional[int] = None) -> dict:
    """Invoice status breakdown: draft, open, paid, overdue counts and amounts.

    Args:
        fiscal_year: Year to summarise. Defaults to current year.

    Returns:
        Dict with status segments matching Billy's insight format.
    """
    year = fiscal_year or date.today().year
    s = await get_invoice_summary(year)
    return {
        "fiscalYear": year,
        "currency": "EUR",
        "segments": [
            {
                "label": "Draft",
                "count": s["draft"]["count"],
                "amount": s["draft"]["amount"],
            },
            {
                "label": "Open",
                "count": s["open"]["count"],
                "amount": s["open"]["amount"],
            },
            {
                "label": "Paid",
                "count": s["paid"]["count"],
                "amount": s["paid"]["amount"],
            },
            {
                "label": "Overdue",
                "count": s["overdue"]["count"],
                "amount": s["overdue"]["amount"],
            },
        ],
    }


async def get_insight_top_customers(
    fiscal_year: Optional[int] = None,
    limit: int = 10,
) -> dict:
    """Top customers ranked by total invoiced amount for the fiscal year.

    Args:
        fiscal_year: Year to analyse. Defaults to current year.
        limit: Number of top customers to return. Defaults to 10.

    Returns:
        Dict with ranked customer rows.
    """
    year = fiscal_year or date.today().year
    invoices = await _fetch_all_invoices_for_year(year)

    totals: dict[str, dict] = {}
    for inv in invoices:
        cid = inv.get("contact_id") or inv.get("customer_name", "unknown")
        name = inv.get("customer_name", cid)
        if cid not in totals:
            totals[cid] = {"name": name, "invoiced": 0.0, "paid": 0.0}
        totals[cid]["invoiced"] += inv["gross_amount"]
        if inv["state"] == "paid":
            totals[cid]["paid"] += inv["gross_amount"]

    ranked = sorted(totals.values(), key=lambda r: r["invoiced"], reverse=True)[:limit]
    return {
        "fiscalYear": year,
        "currency": "EUR",
        "rows": [
            {
                "rank": i + 1,
                "name": r["name"],
                "invoiced": round(r["invoiced"], 2),
                "paid": round(r["paid"], 2),
                "outstanding": round(r["invoiced"] - r["paid"], 2),
            }
            for i, r in enumerate(ranked)
        ],
    }


async def get_insight_aging_report(contact_id: Optional[str] = None) -> dict:
    """Unpaid open invoices bucketed by days overdue.

    Args:
        contact_id: Optionally restrict to a single contact.

    Returns:
        Dict with aging buckets matching Billy's insight format.
    """
    today_dt = date.today()
    today_str = today_dt.isoformat()

    kwargs: dict = {"limit": 500, "offset": 0, "state": "open"}
    if contact_id:
        kwargs["contact_id"] = contact_id

    result = await list_invoices(**kwargs)
    invoices = result.get("invoices", [])

    bucket_order = ["Current", "1–30 days", "31–60 days", "61–90 days", "90+ days"]
    buckets: dict[str, list] = {k: [] for k in bucket_order}

    for inv in invoices:
        due = inv.get("due_date") or today_str
        try:
            days_over = max(0, (today_dt - date.fromisoformat(due)).days)
        except ValueError:
            days_over = 0

        if days_over == 0:
            bucket = "Current"
        elif days_over <= 30:
            bucket = "1–30 days"
        elif days_over <= 60:
            bucket = "31–60 days"
        elif days_over <= 90:
            bucket = "61–90 days"
        else:
            bucket = "90+ days"

        buckets[bucket].append(
            {
                "invoiceNo": inv.get("invoice_no"),
                "customer": inv.get("customer_name"),
                "dueDate": due,
                "amount": round(inv["gross_amount"], 2),
                "daysOverdue": days_over,
            }
        )

    result_buckets = [
        {
            "label": label,
            "totalAmount": round(sum(i["amount"] for i in items), 2),
            "invoices": items,
        }
        for label, items in buckets.items()
        if items
    ]

    return {"currency": "EUR", "asOf": today_str, "buckets": result_buckets}
