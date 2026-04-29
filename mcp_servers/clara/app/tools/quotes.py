"""Quote tools — sevdesk /Offer API."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal, Optional

import httpx
from pydantic import BaseModel

from app.client import get_client
from app.tools.invoices import InvoiceLine, _normalize_date, _sevdesk_date

_OFFER_STATUS = {"100": "draft", "200": "open", "750": "declined", "1000": "accepted"}
_STATUS_TO_CODE = {v: k for k, v in _OFFER_STATUS.items()}

_UNITY_ID = "1"  # sevdesk unity id 1 = Stück (pcs)


class QuoteLine(BaseModel):
    name: str
    quantity: float = 1.0
    unit_price: float
    tax_rate: float = 19.0


def _normalize_offer(o: dict) -> dict:
    status_code = str(o.get("status") or "100")
    contact = o.get("contact") or {}
    return {
        "id": o.get("id"),
        "quote_no": o.get("offerNumber"),
        "contact_id": contact.get("id") if isinstance(contact, dict) else None,
        "customer_name": o.get("contactName")
        or (contact.get("name") if isinstance(contact, dict) else None),
        "entry_date": _normalize_date(o.get("offerDate")),
        "expiry_date": _normalize_date(o.get("validUntil")),
        "state": _OFFER_STATUS.get(status_code, status_code),
        "amount": float(o.get("sumNet") or 0),
        "tax": float(o.get("sumTax") or 0),
        "gross_amount": float(o.get("sumGross") or 0),
        "currency": o.get("currency", "EUR"),
    }


async def list_quotes(
    limit: int = 50,
    offset: int = 0,
    status: Optional[Literal["draft", "open", "declined", "accepted"]] = None,
    contact_id: Optional[str] = None,
) -> dict:
    """Lists quotes (offers) from sevdesk.

    Args:
        limit: Max records to return. Defaults to 50.
        offset: Pagination offset. Defaults to 0.
        status: Filter by state — 'draft', 'open', 'declined', 'accepted'.
        contact_id: Filter by customer/contact ID.

    Returns:
        Dict with total, offset, and a list of quote records.
    """
    params: dict = {"limit": limit, "offset": offset}
    if status and status in _STATUS_TO_CODE:
        params["status"] = _STATUS_TO_CODE[status]
    if contact_id:
        params["contact[id]"] = contact_id
        params["contact[objectName]"] = "Contact"

    try:
        resp = await get_client().get("/Offer", params=params)
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
        "quotes": [_normalize_offer(o) for o in objects],
    }


async def _get_quote_with_lines(quote_id: str) -> dict:
    """Internal: fetch a single offer with its line positions."""
    try:
        resp = await get_client().get(f"/Offer/{quote_id}")
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": f"Quote '{quote_id}' not found."}
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or []
    if not objects:
        return {"error": f"Quote '{quote_id}' not found."}
    o = objects[0] if isinstance(objects, list) else objects
    result = _normalize_offer(o)

    try:
        pos_resp = await get_client().get(
            "/OfferPos",
            params={"offer[id]": quote_id, "offer[objectName]": "Offer", "limit": 100},
        )
        pos_resp.raise_for_status()
        pos_data = pos_resp.json()
        positions = pos_data.get("objects") or []
        result["lines"] = [
            {
                "id": p.get("id"),
                "name": p.get("name"),
                "quantity": float(p.get("quantity") or 1),
                "unit_price": float(p.get("price") or 0),
                "tax_rate": float(p.get("taxRate") or 0),
            }
            for p in positions
        ]
    except (httpx.HTTPStatusError, httpx.RequestError):
        result["lines"] = []

    return result


async def create_quote(
    contact_id: str,
    lines: list[QuoteLine],
    entry_date: Optional[str] = None,
    expiry_days: int = 30,
    currency: str = "EUR",
) -> dict:
    """Creates a new quote (offer) in sevdesk.

    Args:
        contact_id: The sevdesk contact ID to quote.
        lines: Quote line items — each requires name, quantity, unit_price, and tax_rate.
        entry_date: Quote date (YYYY-MM-DD). Defaults to today.
        expiry_days: Days until the quote expires. Defaults to 30.
        currency: Currency code. Defaults to 'EUR'.

    Returns:
        The newly created quote record, or an error dict.
    """
    q_date = date.fromisoformat(entry_date) if entry_date else date.today()
    expiry = q_date + timedelta(days=expiry_days)

    offer_positions = [
        {
            "objectName": "OfferPos",
            "mapAll": True,
            "quantity": ln.quantity,
            "price": ln.unit_price,
            "name": ln.name,
            "unity": {"id": _UNITY_ID, "objectName": "Unity"},
            "taxRate": ln.tax_rate,
            "positionNumber": i + 1,
        }
        for i, ln in enumerate(lines)
    ]

    payload = {
        "offer": {
            "objectName": "Offer",
            "mapAll": True,
            "contact": {"id": contact_id, "objectName": "Contact"},
            "offerDate": _sevdesk_date(q_date.isoformat()),
            "validUntil": _sevdesk_date(expiry.isoformat()),
            "status": "100",
            "currency": currency,
            "header": "Angebot",
            "smallSettlement": False,
        },
        "offerPosSave": offer_positions,
    }

    try:
        resp = await get_client().post("/Offer/Factory/saveOffer", json=payload)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or {}
    offer = objects.get("offer") if isinstance(objects, dict) else {}
    if isinstance(offer, dict) and offer:
        return _normalize_offer(offer)
    return {
        "error": "Offer created but response shape unexpected.",
        "raw": str(objects)[:300],
    }


async def edit_quote(
    quote_id: str,
    status: Optional[Literal["draft", "open", "declined", "accepted"]] = None,
    expiry_days: Optional[int] = None,
) -> dict:
    """Updates an existing quote's status or expiry date.

    Args:
        quote_id: The sevdesk offer ID to update.
        status: Updated state — 'draft', 'open', 'declined', 'accepted'.
        expiry_days: New expiry in days from today.

    Returns:
        The updated quote record, or an error dict.
    """
    payload: dict = {"objectName": "Offer"}
    if status and status in _STATUS_TO_CODE:
        payload["status"] = _STATUS_TO_CODE[status]
    if expiry_days is not None:
        payload["validUntil"] = _sevdesk_date(
            (date.today() + timedelta(days=expiry_days)).isoformat()
        )

    if len(payload) == 1:
        return {"error": "No fields provided to update."}

    try:
        resp = await get_client().put(f"/Offer/{quote_id}", json=payload)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": f"Quote '{quote_id}' not found."}
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or {}
    o = objects if isinstance(objects, dict) else (objects[0] if objects else {})
    return _normalize_offer(o)


async def get_quote_conversion_stats(year: Optional[int] = None) -> dict:
    """Quote pipeline health: counts and conversion rate by status.

    Args:
        year: Calendar year filter. Defaults to current year.

    Returns:
        Dict with year, total, draft, open, accepted, declined, conversion_rate.
    """
    target_year = year or date.today().year

    try:
        resp = await get_client().get("/Offer", params={"limit": 500, "offset": 0})
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        return {"error": str(e)}

    objects = data.get("objects") or []
    counts: dict[str, int] = {"100": 0, "200": 0, "750": 0, "1000": 0}
    for o in objects:
        offer_date = _normalize_date(o.get("offerDate"))
        if offer_date and not offer_date.startswith(str(target_year)):
            continue
        code = str(o.get("status") or "100")
        counts[code] = counts.get(code, 0) + 1

    total = sum(counts.values())
    accepted = counts.get("1000", 0)
    declined = counts.get("750", 0)
    open_sent = accepted + declined + counts.get("200", 0)
    conversion_rate = round(accepted / open_sent, 3) if open_sent > 0 else 0.0

    return {
        "year": target_year,
        "total": total,
        "draft": counts.get("100", 0),
        "open": counts.get("200", 0),
        "accepted": accepted,
        "declined": declined,
        "conversion_rate": conversion_rate,
    }


async def create_invoice_from_quote(quote_id: str) -> dict:
    """Converts an accepted quote into an invoice in sevdesk.

    Fetches the quote's line items and creates a new invoice. Sets the offer
    status to 'declined' (closed) in sevdesk.

    Args:
        quote_id: The sevdesk offer ID to convert (must be in 'accepted' state).

    Returns:
        The newly created invoice record, or an error dict.
    """
    from app.tools.invoices import create_invoice

    quote = await _get_quote_with_lines(quote_id)
    if "error" in quote:
        return quote
    if quote.get("state") != "accepted":
        return {
            "error": (
                f"Quote {quote_id} is in '{quote.get('state')}' state. "
                "Only accepted quotes can be converted to invoices."
            )
        }

    lines_data = quote.get("lines") or []
    if not lines_data:
        return {"error": "Quote has no line items — cannot create invoice."}

    invoice_lines = [
        InvoiceLine(
            name=ln.get("name", "Service"),
            quantity=ln.get("quantity", 1.0),
            unit_price=ln.get("unit_price", 0.0),
            tax_rate=ln.get("tax_rate", 19.0),
        )
        for ln in lines_data
    ]

    invoice = await create_invoice(
        contact_id=quote["contact_id"],
        lines=invoice_lines,
        currency=quote.get("currency", "EUR"),
    )

    # Mark offer as closed (750) — best-effort
    try:
        await get_client().put(
            f"/Offer/{quote_id}", json={"objectName": "Offer", "status": "750"}
        )
    except Exception:
        pass

    if "error" not in invoice:
        invoice["source_quote_id"] = quote_id
    return invoice
