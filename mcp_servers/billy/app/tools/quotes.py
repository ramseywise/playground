"""Quote stub tools for the Billy MCP server."""

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from app.db import get_conn, next_id

_SORT_COLS = {"entry_date", "gross_amount"}


def _fetch_quote_lines(conn, quote_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM quote_lines WHERE quote_id = ? ORDER BY rowid",
        (quote_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_quotes(
    page: int = 1,
    page_size: int = 50,
    states: Optional[list[str]] = None,
    contact_id: Optional[str] = None,
    sort_property: str = "entry_date",
    sort_direction: str = "DESC",
) -> dict:
    """Lists quotes from the accounting system.

    Args:
        page: Page number (1-based). Defaults to 1.
        page_size: Items per page. Defaults to 50.
        states: Filter by quote states: 'open', 'accepted', 'declined', 'invoiced', 'closed'.
        contact_id: Filter by customer/contact ID.
        sort_property: Sort field — 'entry_date' or 'gross_amount'. Defaults to 'entry_date'.
        sort_direction: 'ASC' or 'DESC'. Defaults to 'DESC'.

    Returns:
        Dict with total, page, page_count, and a list of quote records.
    """
    col = sort_property if sort_property in _SORT_COLS else "entry_date"
    direction = "DESC" if sort_direction.upper() == "DESC" else "ASC"

    conditions: list[str] = []
    params: list = []

    if states:
        placeholders = ",".join("?" * len(states))
        conditions.append(f"state IN ({placeholders})")
        params.extend(states)
    if contact_id:
        conditions.append("contact_id = ?")
        params.append(contact_id)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_conn() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM quotes {where}", params).fetchone()[
            0
        ]
        offset = (page - 1) * page_size
        rows = conn.execute(
            f"SELECT * FROM quotes {where} ORDER BY {col} {direction} LIMIT ? OFFSET ?",
            params + [page_size, offset],
        ).fetchall()

    return {
        "total": total,
        "page": page,
        "page_count": max(1, (total + page_size - 1) // page_size),
        "quotes": [
            {
                "id": dict(r)["id"],
                "quote_no": dict(r)["quote_no"],
                "customer_name": dict(r)["customer_name"],
                "entry_date": dict(r)["entry_date"],
                "expiry_date": dict(r)["expiry_date"],
                "state": dict(r)["state"],
                "gross_amount": dict(r)["gross_amount"],
                "currency": dict(r)["currency"],
            }
            for r in rows
        ],
    }


def create_quote(
    contact_id: str,
    lines: list[dict],
    entry_date: Optional[str] = None,
    expiry_days: int = 30,
    currency_id: str = "DKK",
) -> dict:
    """Creates a new quote in the accounting system.

    Args:
        contact_id: The customer/contact ID to quote.
        lines: Quote line items. Each item requires product_id, quantity, and unit_price.
               description is optional.
        entry_date: Quote date (YYYY-MM-DD). Defaults to today.
        expiry_days: Days until the quote expires. Defaults to 30.
        currency_id: Currency code. Defaults to 'DKK'.

    Returns:
        The newly created quote record with lines.
    """
    q_date = entry_date or date.today().isoformat()
    expiry = (date.fromisoformat(q_date) + timedelta(days=expiry_days)).isoformat()
    created_time = datetime.now(timezone.utc).isoformat()

    quote_lines: list[dict] = []
    for i, ln in enumerate(lines):
        qty = float(ln.get("quantity", 1))
        price = float(ln.get("unit_price", 0))
        amount = qty * price
        tax = amount * 0.25
        quote_lines.append(
            {
                "product_id": ln.get("product_id", ""),
                "description": ln.get("description", ""),
                "quantity": qty,
                "unit_price": price,
                "amount": amount,
                "tax": tax,
                "_idx": i,
            }
        )

    total_amount = sum(ln["amount"] for ln in quote_lines)
    total_tax = sum(ln["tax"] for ln in quote_lines)
    gross = total_amount + total_tax

    with get_conn() as conn:
        n = next_id(conn, "quote")
        q_id = f"quo_{n:03d}"
        q_no = f"Q-{date.today().year}-{n:03d}"

        customer_row = conn.execute(
            "SELECT name FROM customers WHERE id = ?", (contact_id,)
        ).fetchone()
        customer_name = customer_row["name"] if customer_row else contact_id

        conn.execute(
            """INSERT INTO quotes
               (id, quote_no, contact_id, customer_name, entry_date, expiry_date,
                state, amount, tax, gross_amount, currency, created_time)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                q_id,
                q_no,
                contact_id,
                customer_name,
                q_date,
                expiry,
                "open",
                total_amount,
                total_tax,
                gross,
                currency_id,
                created_time,
            ),
        )

        for i, ln in enumerate(quote_lines):
            conn.execute(
                """INSERT INTO quote_lines
                   (id, quote_id, product_id, description, quantity,
                    unit_price, amount, tax)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    f"qline_{q_id}_{i}",
                    q_id,
                    ln["product_id"],
                    ln["description"],
                    ln["quantity"],
                    ln["unit_price"],
                    ln["amount"],
                    ln["tax"],
                ),
            )

    return {
        "id": q_id,
        "quote_no": q_no,
        "contact_id": contact_id,
        "customer_name": customer_name,
        "entry_date": q_date,
        "expiry_date": expiry,
        "state": "open",
        "amount": total_amount,
        "tax": total_tax,
        "gross_amount": gross,
        "currency": currency_id,
        "created_time": created_time,
        "lines": [
            {
                "id": f"qline_{q_id}_{ln['_idx']}",
                "product_id": ln["product_id"],
                "description": ln["description"],
                "quantity": ln["quantity"],
                "unit_price": ln["unit_price"],
                "amount": ln["amount"],
                "tax": ln["tax"],
            }
            for ln in quote_lines
        ],
    }


def edit_quote(
    quote_id: str,
    contact_id: Optional[str] = None,
    expiry_days: Optional[int] = None,
    state: Optional[str] = None,
    lines: Optional[list[dict]] = None,
) -> dict:
    """Updates an existing quote. Works on 'open' or 'accepted' quotes.

    Can update the customer, expiry, state, and line items. Invoiced or closed
    quotes cannot be edited.

    Args:
        quote_id: The ID of the quote to update.
        contact_id: Updated customer/contact ID.
        expiry_days: New expiry in days from today (replaces expiry_date).
        state: Set state — 'open', 'accepted', or 'declined'.
        lines: Replacement line items. Each requires product_id, quantity, unit_price.
               Omit to keep existing lines.

    Returns:
        The updated quote record with lines, or an error dict.
    """
    _TERMINAL = {"invoiced", "closed"}
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM quotes WHERE id = ?", (quote_id,)).fetchone()
        if not row:
            return {"error": f"Quote '{quote_id}' not found."}
        quote = dict(row)
        if quote["state"] in _TERMINAL:
            return {
                "error": (
                    f"Quote {quote_id} is in '{quote['state']}' state and cannot be edited."
                )
            }

        updates: list[str] = []
        uparams: list = []
        if contact_id is not None:
            updates.append("contact_id = ?")
            uparams.append(contact_id)
        if expiry_days is not None:
            new_expiry = (date.today() + timedelta(days=expiry_days)).isoformat()
            updates.append("expiry_date = ?")
            uparams.append(new_expiry)
        if state is not None:
            updates.append("state = ?")
            uparams.append(state)

        if updates:
            conn.execute(
                f"UPDATE quotes SET {', '.join(updates)} WHERE id = ?",
                uparams + [quote_id],
            )

        if lines is not None:
            conn.execute("DELETE FROM quote_lines WHERE quote_id = ?", (quote_id,))
            new_lines: list[dict] = []
            for i, ln in enumerate(lines):
                qty = float(ln.get("quantity", 1))
                price = float(ln.get("unit_price", 0))
                amount = qty * price
                tax = amount * 0.25
                line_id = f"qline_{quote_id}_{i}"
                conn.execute(
                    """INSERT INTO quote_lines
                       (id, quote_id, product_id, description, quantity, unit_price, amount, tax)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        line_id,
                        quote_id,
                        ln.get("product_id", ""),
                        ln.get("description", ""),
                        qty,
                        price,
                        amount,
                        tax,
                    ),
                )
                new_lines.append(
                    {
                        "id": line_id,
                        "quantity": qty,
                        "unit_price": price,
                        "amount": amount,
                        "tax": tax,
                    }
                )

            total_amount = sum(nl["amount"] for nl in new_lines)
            total_tax = sum(nl["tax"] for nl in new_lines)
            gross = total_amount + total_tax
            conn.execute(
                "UPDATE quotes SET amount=?, tax=?, gross_amount=? WHERE id=?",
                (total_amount, total_tax, gross, quote_id),
            )

        updated = dict(
            conn.execute("SELECT * FROM quotes WHERE id = ?", (quote_id,)).fetchone()
        )
        final_lines = _fetch_quote_lines(conn, quote_id)

    return {
        "id": updated["id"],
        "quote_no": updated["quote_no"],
        "contact_id": updated["contact_id"],
        "entry_date": updated["entry_date"],
        "expiry_date": updated["expiry_date"],
        "state": updated["state"],
        "amount": updated["amount"],
        "tax": updated["tax"],
        "gross_amount": updated["gross_amount"],
        "currency": updated["currency"],
        "lines": [
            {
                "id": ln["id"],
                "product_id": ln["product_id"],
                "description": ln["description"],
                "quantity": ln["quantity"],
                "unit_price": ln["unit_price"],
                "amount": ln["amount"],
                "tax": ln["tax"],
            }
            for ln in final_lines
        ],
    }


def get_quote_conversion_stats(year: Optional[int] = None) -> dict:
    """Quote pipeline health: sent, accepted, declined counts and conversion rate.

    'Sent' includes all non-open quotes (accepted, declined, invoiced, closed).
    Conversion rate = accepted / sent (where sent > 0).

    Args:
        year: Fiscal year to filter by. Defaults to current year.

    Returns:
        Dict with year, total, sent, accepted, declined, invoiced, conversion_rate.
    """
    target_year = year or date.today().year
    year_prefix = f"{target_year}-%"

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT state, COUNT(*) AS cnt FROM quotes WHERE entry_date LIKE ? GROUP BY state",
            (year_prefix,),
        ).fetchall()

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["state"]] = r["cnt"]

    total = sum(counts.values())
    accepted = counts.get("accepted", 0)
    declined = counts.get("declined", 0)
    invoiced = counts.get("invoiced", 0)
    closed = counts.get("closed", 0)
    sent = accepted + declined + invoiced + closed
    conversion_rate = round(accepted / sent, 3) if sent > 0 else 0.0

    return {
        "year": target_year,
        "total": total,
        "open": counts.get("open", 0),
        "sent": sent,
        "accepted": accepted,
        "declined": declined,
        "invoiced": invoiced,
        "closed": closed,
        "conversion_rate": conversion_rate,
    }


def create_invoice_from_quote(quote_id: str) -> dict:
    """Converts an accepted quote into an invoice.

    Copies the quote's line items onto a new approved invoice. The quote state
    is set to 'invoiced'.

    Args:
        quote_id: The ID of the quote to convert (must be in 'accepted' state).

    Returns:
        The newly created invoice record, or an error dict.
    """
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM quotes WHERE id = ?", (quote_id,)).fetchone()
        if not row:
            return {"error": f"Quote '{quote_id}' not found."}
        quote = dict(row)
        if quote["state"] != "accepted":
            return {
                "error": (
                    f"Quote {quote_id} is in '{quote['state']}' state. "
                    "Only accepted quotes can be converted to invoices."
                )
            }
        line_rows = _fetch_quote_lines(conn, quote_id)

    from app.tools.invoices import InvoiceLine, create_invoice

    invoice_lines = [
        InvoiceLine(
            product_id=ln["product_id"] or "",
            quantity=ln["quantity"],
            unit_price=ln["unit_price"],
            description=ln.get("description"),
        )
        for ln in line_rows
    ]

    invoice = create_invoice(
        contact_id=quote["contact_id"],
        lines=invoice_lines,
        currency_id=quote["currency"],
    )

    with get_conn() as conn:
        conn.execute("UPDATE quotes SET state = 'invoiced' WHERE id = ?", (quote_id,))

    invoice["source_quote_id"] = quote_id
    return invoice
