"""Cross-domain insight tools for the Billy MCP server.

These tools join data across invoices, expenses, and banking tables to answer
profitability, break-even, concentration, and anomaly detection questions.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from app.db import get_conn


def get_net_margin(
    year: Optional[int] = None,
    period: Optional[str] = None,
) -> dict:
    """Net margin: approved invoice revenue minus total expenses for a period.

    Revenue = approved invoice amounts (excl. VAT).
    Total costs = all expense amounts (excl. VAT).
    Net margin % = (revenue − total_costs) / revenue × 100.
    When neither year nor period is given, returns all-time totals.

    Args:
        year: Calendar year, e.g. 2024.
        period: Quarter ('2024-Q1'…'2024-Q4') or month ('2024-01'…'2024-12').
                Overrides year if both are provided.

    Returns:
        Dict with revenue, total_costs, net_profit, net_margin_pct, currency.
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

    inv_where = "WHERE " + " AND ".join(inv_conditions)
    exp_where = ("WHERE " + " AND ".join(exp_conditions)) if exp_conditions else ""

    with get_conn() as conn:
        revenue = conn.execute(
            f"SELECT COALESCE(SUM(amount), 0) FROM invoices {inv_where}", inv_params
        ).fetchone()[0]
        total_costs = conn.execute(
            f"SELECT COALESCE(SUM(amount), 0) FROM expenses {exp_where}", exp_params
        ).fetchone()[0]

    net_profit = revenue - total_costs
    net_margin_pct = round(net_profit / revenue * 100, 1) if revenue > 0 else 0.0

    return {
        "revenue": round(revenue, 2),
        "total_costs": round(total_costs, 2),
        "net_profit": round(net_profit, 2),
        "net_margin_pct": net_margin_pct,
        "currency": "DKK",
    }


def get_margin_by_product(year: Optional[int] = None) -> dict:
    """Revenue and estimated gross margin per product for a given year.

    Revenue per product is derived from approved invoice line items.
    COGS is allocated proportionally from total expenses for the same period
    because expenses are not linked to specific products in the current schema.

    Args:
        year: Calendar year filter. Defaults to current year.

    Returns:
        Dict with year, currency, note, and products list of
        {product_id, product_name, revenue, qty_sold, cogs, gross_profit, margin_pct}.
    """
    target_year = year or date.today().year
    year_str = str(target_year)

    with get_conn() as conn:
        product_rows = conn.execute(
            """
            SELECT il.product_id,
                   COALESCE(p.name, il.description, il.product_id) AS product_name,
                   COALESCE(SUM(il.amount), 0) AS revenue,
                   COALESCE(SUM(il.quantity), 0) AS qty_sold
            FROM invoice_lines il
            JOIN invoices i ON i.id = il.invoice_id
            LEFT JOIN products p ON p.id = il.product_id
            WHERE i.state = 'approved'
              AND strftime('%Y', i.entry_date) = ?
            GROUP BY il.product_id
            ORDER BY revenue DESC
            """,
            (year_str,),
        ).fetchall()

        total_expenses = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM expenses WHERE strftime('%Y', date) = ?",
            (year_str,),
        ).fetchone()[0]

    total_revenue = sum(r["revenue"] for r in product_rows)

    products = []
    for r in product_rows:
        rev = r["revenue"]
        share = rev / total_revenue if total_revenue > 0 else 0.0
        cogs = round(total_expenses * share, 2)
        gross_profit = round(rev - cogs, 2)
        margin_pct = round(gross_profit / rev * 100, 1) if rev > 0 else 0.0
        products.append(
            {
                "product_id": r["product_id"],
                "product_name": r["product_name"],
                "revenue": round(rev, 2),
                "qty_sold": round(r["qty_sold"], 2),
                "cogs": cogs,
                "gross_profit": gross_profit,
                "margin_pct": margin_pct,
            }
        )

    return {
        "year": target_year,
        "currency": "DKK",
        "note": "COGS allocated proportionally from total expenses (no per-product cost data).",
        "products": products,
    }


def get_customer_concentration(year: Optional[int] = None) -> dict:
    """Customer revenue concentration: top-1%, top-3%, and Herfindahl-Hirschman Index.

    HHI measures revenue concentration across customers.
    HHI < 1500 = low concentration; 1500–2500 = moderate; > 2500 = high.
    Top-N percentages show how much of total revenue the largest customers hold.

    Args:
        year: Calendar year filter. Defaults to current year.

    Returns:
        Dict with top_1_pct, top_3_pct, hhi, total_customers, currency,
        and top_customers list (up to 5) of {customer_id, name, revenue, share_pct}.
    """
    target_year = year or date.today().year
    year_prefix = f"{target_year}-%"

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT contact_id, customer_name,
                   COALESCE(SUM(amount), 0) AS revenue
            FROM invoices
            WHERE state = 'approved' AND entry_date LIKE ?
            GROUP BY contact_id
            ORDER BY revenue DESC
            """,
            (year_prefix,),
        ).fetchall()

    if not rows:
        return {
            "year": target_year,
            "top_1_pct": 0.0,
            "top_3_pct": 0.0,
            "hhi": 0.0,
            "total_customers": 0,
            "currency": "DKK",
            "top_customers": [],
        }

    total_revenue = sum(r["revenue"] for r in rows)
    shares = (
        [r["revenue"] / total_revenue for r in rows]
        if total_revenue > 0
        else [0.0] * len(rows)
    )
    hhi = round(sum(s**2 for s in shares) * 10000, 1)
    top_1_pct = round(shares[0] * 100, 1) if shares else 0.0
    top_3_pct = round(sum(shares[:3]) * 100, 1)

    top_customers = [
        {
            "customer_id": rows[i]["contact_id"],
            "name": rows[i]["customer_name"] or rows[i]["contact_id"],
            "revenue": round(rows[i]["revenue"], 2),
            "share_pct": round(shares[i] * 100, 1),
        }
        for i in range(min(5, len(rows)))
    ]

    return {
        "year": target_year,
        "top_1_pct": top_1_pct,
        "top_3_pct": top_3_pct,
        "hhi": hhi,
        "total_customers": len(rows),
        "currency": "DKK",
        "top_customers": top_customers,
    }


def get_dso_trend(months: int = 6) -> dict:
    """Monthly DSO trend for the past N months.

    DSO is approximated as the average payment-terms days
    (julianday(due_date) − julianday(entry_date)) on paid invoices per month.
    Months with no paid invoices have avg_dso=null.

    Args:
        months: Number of months to look back including current month. Range 1–24.

    Returns:
        Dict with trend list of {month, avg_dso, invoice_count} and months_requested.
    """
    months = max(1, min(months, 24))
    today = date.today()

    month_labels: list[str] = []
    for i in range(months - 1, -1, -1):
        raw = today.month - 1 - i
        yr = today.year + raw // 12
        mo = raw % 12 + 1
        month_labels.append(f"{yr}-{mo:02d}")

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT strftime('%Y-%m', entry_date) AS month,
                   AVG(julianday(due_date) - julianday(entry_date)) AS avg_dso,
                   COUNT(*) AS invoice_count
            FROM invoices
            WHERE is_paid = 1
              AND strftime('%Y-%m', entry_date) >= ?
            GROUP BY month
            ORDER BY month
            """,
            (month_labels[0],),
        ).fetchall()

    dso_by_month = {
        r["month"]: {
            "avg_dso": round(r["avg_dso"], 1),
            "invoice_count": r["invoice_count"],
        }
        for r in rows
    }

    trend = [
        {
            "month": label,
            "avg_dso": dso_by_month.get(label, {}).get("avg_dso", None),
            "invoice_count": dso_by_month.get(label, {}).get("invoice_count", 0),
        }
        for label in month_labels
    ]

    return {"trend": trend, "months_requested": months}


def get_break_even_estimate() -> dict:
    """Monthly break-even revenue estimate based on fixed and variable cost structure.

    Fixed costs: monthly average of expenses where is_fixed = True.
    Variable rate: total variable expenses as a fraction of total approved revenue.
    Break-even = monthly_fixed_costs / (1 − variable_rate).
    Returns break_even_revenue=null if variable_rate ≥ 1.0 (costs exceed revenue).

    Returns:
        Dict with fixed_costs (monthly DKK), variable_rate (0.0–1.0+),
        break_even_revenue (DKK or null), currency, and an explanatory note.
    """
    with get_conn() as conn:
        fixed_row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS total_fixed,
                   COUNT(DISTINCT strftime('%Y-%m', date)) AS num_months
            FROM expenses WHERE is_fixed = 1
            """
        ).fetchone()

        total_variable = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM expenses WHERE is_fixed = 0"
        ).fetchone()[0]

        total_revenue = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM invoices WHERE state = 'approved'"
        ).fetchone()[0]

    total_fixed = fixed_row["total_fixed"]
    num_months = max(fixed_row["num_months"], 1)
    monthly_fixed = round(total_fixed / num_months, 2)

    variable_rate = (
        round(total_variable / total_revenue, 4) if total_revenue > 0 else 0.0
    )

    break_even: Optional[float] = None
    if variable_rate < 1.0:
        break_even = round(monthly_fixed / (1.0 - variable_rate), 2)

    return {
        "fixed_costs": monthly_fixed,
        "variable_rate": variable_rate,
        "break_even_revenue": break_even,
        "currency": "DKK",
        "note": "Monthly break-even. Achieve this revenue to cover all fixed and variable costs.",
    }


def detect_anomaly(
    metric: str,
    period: Optional[str] = None,
) -> dict:
    """Detects statistical anomalies in a monthly metric using Z-score analysis.

    Supported metrics:
      - "revenue": monthly approved invoice revenue
      - "expenses": monthly total expense amounts
      - "overdue_rate": fraction of open approved invoices past due per month
      - "dso": average payment-terms days per month (paid invoices only)

    A month is flagged as anomalous when its absolute Z-score exceeds 1.5.
    Requires ≥ 3 data points for meaningful analysis.

    Args:
        metric: One of 'revenue', 'expenses', 'overdue_rate', 'dso'.
        period: Year string e.g. '2024' to restrict the analysis window.
                Omit to analyse all available data.

    Returns:
        Dict with metric, mean, std_dev, anomalies list of
        {month, value, z_score, direction}, and data_points list.
    """
    metric = metric.lower()
    supported = {"revenue", "expenses", "overdue_rate", "dso"}
    if metric not in supported:
        return {
            "error": f"Unsupported metric '{metric}'. Choose from: {', '.join(sorted(supported))}."
        }

    year_cond_inv = ""
    year_cond_exp = ""
    year_params: list[str] = []

    if period:
        try:
            yr = str(int(str(period).split("-")[0]))
            year_cond_inv = "AND strftime('%Y', entry_date) = ?"
            year_cond_exp = "AND strftime('%Y', date) = ?"
            year_params = [yr]
        except (ValueError, IndexError):
            pass

    today_str = date.today().isoformat()

    with get_conn() as conn:
        if metric == "revenue":
            rows = conn.execute(
                f"""SELECT strftime('%Y-%m', entry_date) AS month,
                           COALESCE(SUM(amount), 0) AS value
                    FROM invoices WHERE state = 'approved' {year_cond_inv}
                    GROUP BY month ORDER BY month""",
                year_params,
            ).fetchall()

        elif metric == "expenses":
            rows = conn.execute(
                f"""SELECT strftime('%Y-%m', date) AS month,
                           COALESCE(SUM(amount), 0) AS value
                    FROM expenses WHERE 1=1 {year_cond_exp}
                    GROUP BY month ORDER BY month""",
                year_params,
            ).fetchall()

        elif metric == "overdue_rate":
            rows = conn.execute(
                f"""SELECT strftime('%Y-%m', entry_date) AS month,
                           CAST(SUM(CASE WHEN due_date < ? AND is_paid = 0 THEN 1 ELSE 0 END) AS REAL)
                           / NULLIF(COUNT(*), 0) AS value
                    FROM invoices WHERE state = 'approved' {year_cond_inv}
                    GROUP BY month ORDER BY month""",
                [today_str] + year_params,
            ).fetchall()

        else:  # dso
            rows = conn.execute(
                f"""SELECT strftime('%Y-%m', entry_date) AS month,
                           AVG(julianday(due_date) - julianday(entry_date)) AS value
                    FROM invoices WHERE is_paid = 1 {year_cond_inv}
                    GROUP BY month ORDER BY month""",
                year_params,
            ).fetchall()

    data_points = [
        {"month": r["month"], "value": round(float(r["value"] or 0.0), 2)} for r in rows
    ]

    if len(data_points) < 3:
        return {
            "metric": metric,
            "anomalies": [],
            "data_points": data_points,
            "note": "Insufficient data for anomaly detection (need ≥ 3 months).",
        }

    values = [p["value"] for p in data_points]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance**0.5

    anomalies = []
    for p in data_points:
        z = (p["value"] - mean) / std if std > 0 else 0.0
        if abs(z) >= 1.5:
            anomalies.append(
                {
                    "month": p["month"],
                    "value": p["value"],
                    "z_score": round(z, 2),
                    "direction": "high" if z > 0 else "low",
                }
            )

    return {
        "metric": metric,
        "mean": round(mean, 2),
        "std_dev": round(std, 2),
        "anomalies": anomalies,
        "data_points": data_points,
    }
