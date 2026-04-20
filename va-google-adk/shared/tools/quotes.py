"""Quote tools for the VA accounting assistant."""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_MOCK_QUOTES: list[dict] = [
    {
        "id": "quo_001",
        "quoteNo": "Q-2024-001",
        "contactId": "cus_001",
        "customerName": "Acme A/S",
        "entryDate": "2024-03-01",
        "expiryDate": "2024-03-31",
        "state": "open",
        "amount": 20000.00,
        "tax": 5000.00,
        "grossAmount": 25000.00,
        "currency": "DKK",
        "lines": [
            {
                "id": "qline_001a",
                "productId": "prod_001",
                "description": "Konsulentydelser Q1",
                "quantity": 20,
                "unitPrice": 1000.00,
                "amount": 20000.00,
                "tax": 5000.00,
            }
        ],
        "createdTime": "2024-03-01T09:00:00Z",
    },
    {
        "id": "quo_002",
        "quoteNo": "Q-2024-002",
        "contactId": "cus_002",
        "customerName": "Nordisk Tech ApS",
        "entryDate": "2024-04-01",
        "expiryDate": "2024-04-30",
        "state": "accepted",
        "amount": 15000.00,
        "tax": 3750.00,
        "grossAmount": 18750.00,
        "currency": "DKK",
        "lines": [
            {
                "id": "qline_002a",
                "productId": "prod_002",
                "description": "Softwarelicens Q2",
                "quantity": 3,
                "unitPrice": 5000.00,
                "amount": 15000.00,
                "tax": 3750.00,
            }
        ],
        "createdTime": "2024-04-01T10:00:00Z",
    },
    {
        "id": "quo_003",
        "quoteNo": "Q-2025-001",
        "contactId": "cus_001",
        "customerName": "Acme A/S",
        "entryDate": "2025-01-10",
        "expiryDate": "2025-02-10",
        "state": "declined",
        "amount": 8000.00,
        "tax": 2000.00,
        "grossAmount": 10000.00,
        "currency": "DKK",
        "lines": [
            {
                "id": "qline_003a",
                "productId": "prod_004",
                "description": "Uddannelsesdage",
                "quantity": 1,
                "unitPrice": 8000.00,
                "amount": 8000.00,
                "tax": 2000.00,
            }
        ],
        "createdTime": "2025-01-10T08:30:00Z",
    },
]

_next_quote_counter = 4


def _find_quote(quote_id: str) -> Optional[dict]:
    return next((q for q in _MOCK_QUOTES if q["id"] == quote_id), None)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def list_quotes(
    page: int = 1,
    page_size: int = 50,
    states: Optional[list] = None,
    contact_id: Optional[str] = None,
    sort_property: str = "entryDate",
    sort_direction: str = "DESC",
) -> dict:
    """Lists quotes from the accounting system.

    Args:
        page: Page number (1-based). Defaults to 1.
        page_size: Items per page. Defaults to 50.
        states: Filter by quote states: 'open', 'accepted', 'declined', 'invoiced', 'closed'.
        contact_id: Filter by customer/contact ID.
        sort_property: Sort field — 'entryDate' or 'grossAmount'. Defaults to 'entryDate'.
        sort_direction: 'ASC' or 'DESC'. Defaults to 'DESC'.

    Returns:
        Dict with total, page, pageCount, and a list of quote records.
    """
    quotes = list(_MOCK_QUOTES)
    if states:
        quotes = [q for q in quotes if q["state"] in states]
    if contact_id:
        quotes = [q for q in quotes if q["contactId"] == contact_id]

    reverse = sort_direction.upper() == "DESC"
    quotes.sort(key=lambda q: q.get(sort_property, ""), reverse=reverse)

    total = len(quotes)
    start = (page - 1) * page_size
    page_quotes = quotes[start : start + page_size]

    return {
        "total": total,
        "page": page,
        "pageCount": max(1, (total + page_size - 1) // page_size),
        "quotes": [
            {
                "id": q["id"],
                "quoteNo": q["quoteNo"],
                "customerName": q["customerName"],
                "entryDate": q["entryDate"],
                "expiryDate": q["expiryDate"],
                "state": q["state"],
                "grossAmount": q["grossAmount"],
                "currency": q["currency"],
            }
            for q in page_quotes
        ],
    }


def create_quote(
    contact_id: str,
    lines: list,
    entry_date: Optional[str] = None,
    expiry_days: int = 30,
    currency_id: str = "DKK",
) -> dict:
    """Creates a new quote in the accounting system.

    Args:
        contact_id: The customer/contact ID to quote.
        lines: Quote line items. Each item requires productId, quantity, and unitPrice.
               description is optional.
        entry_date: Quote date (YYYY-MM-DD). Defaults to today.
        expiry_days: Days until the quote expires. Defaults to 30.
        currency_id: Currency code. Defaults to 'DKK'.

    Returns:
        The newly created quote record with lines.
    """
    global _next_quote_counter
    q_date = entry_date or date.today().isoformat()
    q_id = f"quo_{_next_quote_counter:03d}"
    q_no = f"Q-2026-{_next_quote_counter:03d}"
    _next_quote_counter += 1

    quote_lines = []
    for i, ln in enumerate(lines):
        qty = ln.get("quantity", 1)
        price = ln.get("unitPrice", 0)
        amount = qty * price
        tax = amount * 0.25
        quote_lines.append({
            "id": f"qline_{q_id}_{i}",
            "productId": ln.get("productId", ""),
            "description": ln.get("description", ""),
            "quantity": qty,
            "unitPrice": price,
            "amount": amount,
            "tax": tax,
        })

    total_amount = sum(ln["amount"] for ln in quote_lines)
    total_tax = sum(ln["tax"] for ln in quote_lines)
    gross = total_amount + total_tax
    expiry = (date.fromisoformat(q_date) + timedelta(days=expiry_days)).isoformat()

    new_quote = {
        "id": q_id,
        "quoteNo": q_no,
        "contactId": contact_id,
        "customerName": contact_id,
        "entryDate": q_date,
        "expiryDate": expiry,
        "state": "open",
        "amount": total_amount,
        "tax": total_tax,
        "grossAmount": gross,
        "currency": currency_id,
        "lines": quote_lines,
        "createdTime": datetime.now(timezone.utc).isoformat(),
    }
    _MOCK_QUOTES.append(new_quote)
    return new_quote


def create_invoice_from_quote(quote_id: str) -> dict:
    """Converts an accepted quote into an invoice.

    Copies the quote's line items onto a new approved invoice. The quote state
    is set to 'invoiced'.

    Args:
        quote_id: The ID of the quote to convert (must be in 'accepted' state).

    Returns:
        The newly created invoice record, or an error dict.
    """
    quote = _find_quote(quote_id)
    if not quote:
        return {"error": f"Quote '{quote_id}' not found."}
    if quote["state"] != "accepted":
        return {
            "error": (
                f"Quote {quote_id} is in '{quote['state']}' state. "
                "Only accepted quotes can be converted to invoices."
            )
        }

    # Import lazily to avoid circular dependency
    from .invoices import create_invoice

    lines = [
        {
            "productId": ln["productId"],
            "description": ln["description"],
            "quantity": ln["quantity"],
            "unitPrice": ln["unitPrice"],
        }
        for ln in quote["lines"]
    ]

    invoice = create_invoice(
        contact_id=quote["contactId"],
        lines=lines,
        currency_id=quote["currency"],
    )

    # Mark quote as invoiced
    quote["state"] = "invoiced"
    invoice["sourceQuoteId"] = quote_id
    return invoice
