"""Expense stub tools for the Billy MCP server."""

from __future__ import annotations

from typing import Optional

from app.db import get_conn, next_id

_VALID_CATEGORIES = frozenset(
    [
        "rent",
        "salaries",
        "software",
        "marketing",
        "office",
        "travel",
        "meals",
        "professional_services",
        "utilities",
        "other",
    ]
)

# VAT-exempt categories in Denmark
_VAT_EXEMPT = frozenset({"rent", "salaries"})


def list_expenses(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    category: Optional[str] = None,
    vendor: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> dict:
    """Lists expense records with optional date, category, and vendor filters.

    Returns a paginated list of expenses sorted by date descending.

    Args:
        date_from: Earliest date inclusive (ISO format, e.g. '2024-01-01').
        date_to: Latest date inclusive (ISO format, e.g. '2024-12-31').
        category: Filter by category — rent, salaries, software, marketing,
            office, travel, meals, professional_services, utilities, other.
        vendor: Case-insensitive substring filter on vendor name.
        page: Page number (1-based). Defaults to 1.
        page_size: Items per page. Defaults to 50.

    Returns:
        Dict with total, page, page_count, and list of expense records.
    """
    conditions: list[str] = []
    params: list = []

    if date_from:
        conditions.append("date >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("date <= ?")
        params.append(date_to)
    if category:
        conditions.append("category = ?")
        params.append(category)
    if vendor:
        conditions.append("LOWER(vendor) LIKE LOWER(?)")
        params.append(f"%{vendor}%")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_conn() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM expenses {where}", params
        ).fetchone()[0]
        offset = (page - 1) * page_size
        rows = conn.execute(
            f"SELECT * FROM expenses {where} ORDER BY date DESC, id DESC LIMIT ? OFFSET ?",
            params + [page_size, offset],
        ).fetchall()

    expenses = []
    for r in rows:
        e = dict(r)
        e["is_fixed"] = bool(e["is_fixed"])
        expenses.append(e)

    return {
        "total": total,
        "page": page,
        "page_count": max(1, (total + page_size - 1) // page_size),
        "expenses": expenses,
    }


def get_expense(expense_id: str) -> dict:
    """Gets a single expense record by ID.

    Args:
        expense_id: The ID of the expense to look up.

    Returns:
        Full expense record, or an error dict if not found.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM expenses WHERE id = ?", (expense_id,)
        ).fetchone()
        if not row:
            return {"error": f"Expense '{expense_id}' not found."}
        e = dict(row)
        e["is_fixed"] = bool(e["is_fixed"])
    return e


def create_expense(
    vendor: str,
    amount: float,
    date: str,
    category: str = "other",
    description: Optional[str] = None,
    vat_amount: Optional[float] = None,
    is_fixed: bool = False,
    contact_id: Optional[str] = None,
    currency: str = "DKK",
) -> dict:
    """Records a new expense in the accounting system.

    Amount is excl. VAT. If vat_amount is not provided it is calculated at
    25% for standard categories; rent and salaries are VAT-exempt (0%).

    Args:
        vendor: Name of the vendor or supplier.
        amount: Expense amount excl. VAT.
        date: Expense date in ISO format (YYYY-MM-DD).
        category: Category — rent, salaries, software, marketing, office,
            travel, meals, professional_services, utilities, other.
        description: Optional description or memo.
        vat_amount: VAT amount override. Defaults to 25% (0% for exempt categories).
        is_fixed: True if this is a fixed/recurring cost. Defaults to False.
        contact_id: Optional customer/supplier ID to link this expense.
        currency: Currency code. Defaults to 'DKK'.

    Returns:
        The newly created expense record.
    """
    if category not in _VALID_CATEGORIES:
        category = "other"

    if vat_amount is None:
        vat_amount = 0.0 if category in _VAT_EXEMPT else round(amount * 0.25, 2)

    gross_amount = round(amount + vat_amount, 2)
    created_time = "2026-04-21T10:00:00Z"

    with get_conn() as conn:
        n = next_id(conn, "expense")
        new_id = f"exp_{n:03d}"
        conn.execute(
            """INSERT INTO expenses
               (id, vendor, amount, tax, gross_amount, currency, date, category,
                is_fixed, description, contact_id, status, created_time)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                new_id,
                vendor,
                amount,
                vat_amount,
                gross_amount,
                currency,
                date,
                category,
                int(is_fixed),
                description,
                contact_id,
                "approved",
                created_time,
            ),
        )

    return {
        "id": new_id,
        "vendor": vendor,
        "amount": amount,
        "tax": vat_amount,
        "gross_amount": gross_amount,
        "currency": currency,
        "date": date,
        "category": category,
        "is_fixed": is_fixed,
        "description": description,
        "contact_id": contact_id,
        "status": "approved",
        "created_time": created_time,
    }


def get_expense_summary(
    year: Optional[int] = None,
    period: Optional[str] = None,
) -> dict:
    """Returns total expenses broken down by category for a given period.

    If period is provided (e.g. '2024-Q1' or '2024-01'), filtering is by that
    quarter or month. If year is provided without period, full-year totals are
    returned. If neither is given, all-time summary is returned.

    Args:
        year: Calendar year, e.g. 2024.
        period: Quarter ('2024-Q1' … '2024-Q4') or month ('2024-01' … '2024-12').

    Returns:
        Dict with total, currency, and by_category list of
        {category, amount, percentage}.
    """
    conditions: list[str] = []
    params: list = []

    if period and "Q" in str(period):
        try:
            yr, q_str = str(period).split("-Q")
            q = int(q_str)
            month_start = (q - 1) * 3 + 1
            month_end = q * 3
            conditions.append("strftime('%Y', date) = ?")
            params.append(yr)
            conditions.append("CAST(strftime('%m', date) AS INTEGER) BETWEEN ? AND ?")
            params.extend([month_start, month_end])
        except Exception:
            pass
    elif period:
        conditions.append("strftime('%Y-%m', date) = ?")
        params.append(str(period))
    elif year:
        conditions.append("strftime('%Y', date) = ?")
        params.append(str(year))

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_conn() as conn:
        total = conn.execute(
            f"SELECT COALESCE(SUM(amount), 0) FROM expenses {where}", params
        ).fetchone()[0]

        rows = conn.execute(
            f"""SELECT category, SUM(amount) as cat_total
                FROM expenses {where}
                GROUP BY category
                ORDER BY cat_total DESC""",
            params,
        ).fetchall()

    by_category = []
    for r in rows:
        cat_total = r[1]
        pct = round(cat_total / total * 100, 1) if total > 0 else 0.0
        by_category.append({"category": r[0], "amount": cat_total, "percentage": pct})

    return {"total": total, "currency": "DKK", "by_category": by_category}


def get_vendor_spend(
    vendor: Optional[str] = None,
    year: Optional[int] = None,
) -> dict:
    """Returns total spend per vendor, optionally filtered by name or year.

    Useful for vendor audits — which suppliers cost the most?

    Args:
        vendor: Case-insensitive substring filter on vendor name.
        year: Filter to a specific calendar year.

    Returns:
        Dict with vendors list of {vendor, total, count, currency}.
    """
    conditions: list[str] = []
    params: list = []

    if vendor:
        conditions.append("LOWER(vendor) LIKE LOWER(?)")
        params.append(f"%{vendor}%")
    if year:
        conditions.append("strftime('%Y', date) = ?")
        params.append(str(year))

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT vendor, SUM(amount) as total, COUNT(*) as count
                FROM expenses {where}
                GROUP BY vendor
                ORDER BY total DESC""",
            params,
        ).fetchall()

    return {
        "vendors": [
            {"vendor": r[0], "total": r[1], "count": r[2], "currency": "DKK"}
            for r in rows
        ]
    }


def get_expenses_by_category(year: Optional[int] = None) -> dict:
    """Returns total expenses per category with fixed/variable classification.

    The is_fixed flag distinguishes recurring fixed costs (rent, licences)
    from variable costs (marketing, meals, travel). A category is classified
    as fixed when the majority of its expense records carry is_fixed=True.

    Args:
        year: Calendar year filter. If omitted, returns all-time totals.

    Returns:
        Dict with categories list of {category, total, is_fixed, currency}.
    """
    conditions: list[str] = []
    params: list = []

    if year:
        conditions.append("strftime('%Y', date) = ?")
        params.append(str(year))

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT category,
                       SUM(amount) as total,
                       CASE WHEN SUM(is_fixed) * 2 > COUNT(*) THEN 1 ELSE 0 END as is_fixed
                FROM expenses {where}
                GROUP BY category
                ORDER BY total DESC""",
            params,
        ).fetchall()

    return {
        "categories": [
            {
                "category": r[0],
                "total": r[1],
                "is_fixed": bool(r[2]),
                "currency": "DKK",
            }
            for r in rows
        ]
    }


def get_gross_margin(
    year: Optional[int] = None,
    period: Optional[str] = None,
) -> dict:
    """Returns gross margin by comparing approved invoice revenue against expenses.

    Revenue = sum of approved invoice amounts (excl. VAT).
    COGS    = sum of all expense amounts (excl. VAT) for the same period.
    Gross margin % = (revenue − COGS) / revenue × 100.

    Args:
        year: Calendar year, e.g. 2024.
        period: Quarter ('2024-Q1' … '2024-Q4') or month ('2024-01' … '2024-12').

    Returns:
        Dict with revenue, cogs, gross_profit, gross_margin_pct, currency.
    """
    inv_conditions: list[str] = ["state = 'approved'"]
    exp_conditions: list[str] = []
    inv_params: list = []
    exp_params: list = []

    if period and "Q" in str(period):
        try:
            yr, q_str = str(period).split("-Q")
            q = int(q_str)
            month_start = (q - 1) * 3 + 1
            month_end = q * 3
            inv_conditions.append("strftime('%Y', entry_date) = ?")
            inv_params.append(yr)
            inv_conditions.append(
                "CAST(strftime('%m', entry_date) AS INTEGER) BETWEEN ? AND ?"
            )
            inv_params.extend([month_start, month_end])
            exp_conditions.append("strftime('%Y', date) = ?")
            exp_params.append(yr)
            exp_conditions.append(
                "CAST(strftime('%m', date) AS INTEGER) BETWEEN ? AND ?"
            )
            exp_params.extend([month_start, month_end])
        except Exception:
            pass
    elif period:
        inv_conditions.append("strftime('%Y-%m', entry_date) = ?")
        inv_params.append(str(period))
        exp_conditions.append("strftime('%Y-%m', date) = ?")
        exp_params.append(str(period))
    elif year:
        inv_conditions.append("strftime('%Y', entry_date) = ?")
        inv_params.append(str(year))
        exp_conditions.append("strftime('%Y', date) = ?")
        exp_params.append(str(year))

    inv_where = f"WHERE {' AND '.join(inv_conditions)}"
    exp_where = f"WHERE {' AND '.join(exp_conditions)}" if exp_conditions else ""

    with get_conn() as conn:
        revenue = conn.execute(
            f"SELECT COALESCE(SUM(amount), 0) FROM invoices {inv_where}", inv_params
        ).fetchone()[0]

        cogs = conn.execute(
            f"SELECT COALESCE(SUM(amount), 0) FROM expenses {exp_where}", exp_params
        ).fetchone()[0]

    gross_profit = revenue - cogs
    gross_margin_pct = round(gross_profit / revenue * 100, 1) if revenue > 0 else 0.0

    return {
        "revenue": revenue,
        "cogs": cogs,
        "gross_profit": gross_profit,
        "gross_margin_pct": gross_margin_pct,
        "currency": "DKK",
    }
