"""Cross-domain insight tools — aggregated from invoices, vouchers, and bank accounts."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Optional

import httpx

from app.client import get_client
from app.tools.expenses import _fetch_all_vouchers_for_year
from app.tools.invoices import _fetch_all_invoices_for_year


def _month_in_quarter(date_str: str, month_start: int, month_end: int) -> bool:
    if len(date_str) < 7:
        return False
    try:
        return month_start <= int(date_str[5:7]) <= month_end
    except (ValueError, IndexError):
        return False


async def _load_period(
    year: Optional[int], period: Optional[str]
) -> tuple[list[dict], list[dict], int]:
    """Fetch and optionally filter invoices + vouchers for a period."""
    target_year = year or date.today().year
    invoices = await _fetch_all_invoices_for_year(target_year)
    vouchers = await _fetch_all_vouchers_for_year(target_year)

    if period and "Q" in str(period):
        try:
            _, q_str = str(period).split("-Q")
            q = int(q_str)
            ms, me = (q - 1) * 3 + 1, q * 3
            invoices = [
                i
                for i in invoices
                if _month_in_quarter(i.get("invoice_date", ""), ms, me)
            ]
            vouchers = [
                v for v in vouchers if _month_in_quarter(v.get("date", ""), ms, me)
            ]
        except Exception:
            pass
    elif period:
        p = str(period)
        invoices = [i for i in invoices if (i.get("invoice_date") or "").startswith(p)]
        vouchers = [v for v in vouchers if (v.get("date") or "").startswith(p)]

    return invoices, vouchers, target_year


async def get_net_margin(
    year: Optional[int] = None,
    period: Optional[str] = None,
) -> dict:
    """Net margin: invoice revenue minus total expenses for a period.

    Revenue = open/paid invoice net amounts (excl. VAT).
    Total costs = all voucher net amounts (excl. VAT).
    Net margin % = (revenue − total_costs) / revenue × 100.
    When neither year nor period is given, returns current-year totals.

    Args:
        year: Calendar year, e.g. 2024.
        period: Quarter ('2024-Q1'…'2024-Q4') or month ('2024-01'…'2024-12').

    Returns:
        Dict with revenue, total_costs, net_profit, net_margin_pct, currency.
    """
    invoices, vouchers, _ = await _load_period(year, period)

    revenue = sum(
        float(inv.get("amount") or 0)
        for inv in invoices
        if inv.get("state") in ("open", "partially_paid", "paid")
    )
    total_costs = sum(float(v.get("amount") or 0) for v in vouchers)
    net_profit = revenue - total_costs
    net_margin_pct = round(net_profit / revenue * 100, 1) if revenue > 0 else 0.0

    return {
        "revenue": round(revenue, 2),
        "total_costs": round(total_costs, 2),
        "net_profit": round(net_profit, 2),
        "net_margin_pct": net_margin_pct,
        "currency": "EUR",
    }


async def get_margin_by_product(year: Optional[int] = None) -> dict:
    """Revenue and estimated gross margin per product for a given year.

    Fetches invoice positions for up to 30 invoices in the year and groups by
    product/service name. COGS is allocated proportionally from total expenses.

    Args:
        year: Calendar year filter. Defaults to current year.

    Returns:
        Dict with year, currency, note, and products list of
        {product_name, revenue, qty_sold, cogs, gross_profit, margin_pct}.
    """
    target_year = year or date.today().year
    invoices = await _fetch_all_invoices_for_year(target_year)
    inv_ids = [inv["id"] for inv in invoices if inv.get("id")][:30]

    async def _fetch_positions(inv_id: str) -> list[dict]:
        try:
            resp = await get_client().get(
                "/InvoicePos",
                params={
                    "invoice[id]": inv_id,
                    "invoice[objectName]": "Invoice",
                    "limit": 100,
                },
            )
            resp.raise_for_status()
            return resp.json().get("objects") or []
        except (httpx.HTTPStatusError, httpx.RequestError):
            return []

    position_groups = await asyncio.gather(
        *[_fetch_positions(inv_id) for inv_id in inv_ids]
    )
    all_positions = [pos for group in position_groups for pos in group]

    vouchers = await _fetch_all_vouchers_for_year(target_year)
    total_expenses = sum(float(v.get("amount") or 0) for v in vouchers)

    by_product: dict[str, dict] = {}
    for pos in all_positions:
        name = pos.get("name") or "Unknown"
        qty = float(pos.get("quantity") or 0)
        net = float(pos.get("sumNet") or (qty * float(pos.get("price") or 0)))
        if name not in by_product:
            by_product[name] = {"revenue": 0.0, "qty_sold": 0.0}
        by_product[name]["revenue"] += net
        by_product[name]["qty_sold"] += qty

    total_revenue = sum(p["revenue"] for p in by_product.values())

    products = []
    for name, data in sorted(
        by_product.items(), key=lambda x: x[1]["revenue"], reverse=True
    ):
        rev = data["revenue"]
        share = rev / total_revenue if total_revenue > 0 else 0.0
        cogs = round(total_expenses * share, 2)
        gross_profit = round(rev - cogs, 2)
        margin_pct = round(gross_profit / rev * 100, 1) if rev > 0 else 0.0
        products.append(
            {
                "product_name": name,
                "revenue": round(rev, 2),
                "qty_sold": round(data["qty_sold"], 2),
                "cogs": cogs,
                "gross_profit": gross_profit,
                "margin_pct": margin_pct,
            }
        )

    return {
        "year": target_year,
        "currency": "EUR",
        "note": f"Based on up to 30 invoices for {target_year}. COGS allocated proportionally from total expenses.",
        "products": products,
    }


async def get_customer_concentration(year: Optional[int] = None) -> dict:
    """Customer revenue concentration: top-1%, top-3%, and Herfindahl-Hirschman Index.

    HHI < 1500 = low concentration; 1500–2500 = moderate; > 2500 = high.
    Top-N percentages show how much of total revenue the largest customers hold.

    Args:
        year: Calendar year filter. Defaults to current year.

    Returns:
        Dict with top_1_pct, top_3_pct, hhi, total_customers, currency,
        and top_customers list (up to 5) of {customer_id, name, revenue, share_pct}.
    """
    target_year = year or date.today().year
    invoices = await _fetch_all_invoices_for_year(target_year)

    by_customer: dict[str, dict] = {}
    for inv in invoices:
        if inv.get("state") not in ("open", "partially_paid", "paid"):
            continue
        cid = inv.get("contact_id") or inv.get("customer_name") or "unknown"
        name = inv.get("customer_name") or cid
        if cid not in by_customer:
            by_customer[cid] = {"name": name, "revenue": 0.0}
        by_customer[cid]["revenue"] += float(inv.get("gross_amount") or 0)

    if not by_customer:
        return {
            "year": target_year,
            "top_1_pct": 0.0,
            "top_3_pct": 0.0,
            "hhi": 0.0,
            "total_customers": 0,
            "currency": "EUR",
            "top_customers": [],
        }

    ranked = sorted(by_customer.items(), key=lambda x: x[1]["revenue"], reverse=True)
    total_revenue = sum(d["revenue"] for _, d in ranked)
    shares = (
        [d["revenue"] / total_revenue for _, d in ranked]
        if total_revenue > 0
        else [0.0] * len(ranked)
    )

    hhi = round(sum(s**2 for s in shares) * 10000, 1)
    top_1_pct = round(shares[0] * 100, 1) if shares else 0.0
    top_3_pct = round(sum(shares[:3]) * 100, 1)

    top_customers = [
        {
            "customer_id": cid,
            "name": data["name"],
            "revenue": round(data["revenue"], 2),
            "share_pct": round(share * 100, 1),
        }
        for (cid, data), share in zip(ranked[:5], shares[:5])
    ]

    return {
        "year": target_year,
        "top_1_pct": top_1_pct,
        "top_3_pct": top_3_pct,
        "hhi": hhi,
        "total_customers": len(by_customer),
        "currency": "EUR",
        "top_customers": top_customers,
    }


async def get_dso_trend(months: int = 6) -> dict:
    """Monthly DSO trend for the past N months.

    DSO is approximated as average payment terms (due_date − invoice_date) for
    paid invoices per month. Note: actual payment receipt dates are not exposed
    by the sevdesk API, so payment terms days are used as a proxy.

    Args:
        months: Number of months to look back including current month (1–24).

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

    years = {int(label[:4]) for label in month_labels}
    all_invoices: list[dict] = []
    for yr in years:
        all_invoices.extend(await _fetch_all_invoices_for_year(yr))

    dso_by_month: dict[str, list[float]] = {label: [] for label in month_labels}
    for inv in all_invoices:
        if inv.get("state") != "paid":
            continue
        inv_date = inv.get("invoice_date") or ""
        due_date = inv.get("due_date") or ""
        month = inv_date[:7]
        if month not in dso_by_month or len(inv_date) < 10 or len(due_date) < 10:
            continue
        try:
            delta = (date.fromisoformat(due_date) - date.fromisoformat(inv_date)).days
            dso_by_month[month].append(float(delta))
        except ValueError:
            continue

    trend = [
        {
            "month": label,
            "avg_dso": round(sum(vals) / len(vals), 1) if vals else None,
            "invoice_count": len(vals),
        }
        for label, vals in dso_by_month.items()
    ]

    return {"trend": trend, "months_requested": months}


async def get_break_even_estimate() -> dict:
    """Monthly break-even revenue estimate based on expense structure.

    Note: sevdesk does not distinguish fixed vs variable expenses. All expenses
    are treated as variable. Break-even is the average monthly expense — the
    minimum revenue needed to cover all costs.

    Returns:
        Dict with fixed_costs, variable_rate, break_even_revenue, currency, note.
    """
    today = date.today()
    all_vouchers: list[dict] = []
    for yr in [today.year, today.year - 1]:
        all_vouchers.extend(await _fetch_all_vouchers_for_year(yr))

    invoices = await _fetch_all_invoices_for_year(today.year)
    total_revenue = sum(
        float(inv.get("amount") or 0)
        for inv in invoices
        if inv.get("state") in ("open", "partially_paid", "paid")
    )

    monthly: dict[str, float] = {}
    for v in all_vouchers:
        dt = v.get("date") or ""
        if len(dt) >= 7:
            m = dt[:7]
            monthly[m] = monthly.get(m, 0.0) + float(v.get("amount") or 0)

    avg_monthly = round(sum(monthly.values()) / len(monthly), 2) if monthly else 0.0
    total_variable = sum(float(v.get("amount") or 0) for v in all_vouchers)
    variable_rate = (
        round(total_variable / total_revenue, 4) if total_revenue > 0 else 0.0
    )

    break_even: Optional[float] = None
    if variable_rate < 1.0 and avg_monthly > 0:
        break_even = round(avg_monthly / (1.0 - variable_rate), 2)

    return {
        "fixed_costs": 0.0,
        "variable_rate": variable_rate,
        "break_even_revenue": break_even or avg_monthly,
        "avg_monthly_expenses": avg_monthly,
        "currency": "EUR",
        "note": "sevdesk does not distinguish fixed/variable costs — all expenses are variable. Break-even = minimum monthly revenue to cover all costs.",
    }


async def detect_anomaly(
    metric: str,
    period: Optional[str] = None,
) -> dict:
    """Detects statistical anomalies in a monthly metric using Z-score analysis.

    Supported metrics:
      - 'revenue': monthly invoice net revenue
      - 'expenses': monthly voucher net spend
      - 'overdue_rate': fraction of invoices from each month now overdue
      - 'dso': average payment terms (due − invoice date) per month for paid invoices

    A month is flagged as anomalous when its absolute Z-score exceeds 1.5.
    Requires ≥ 3 data points for meaningful analysis.

    Args:
        metric: One of 'revenue', 'expenses', 'overdue_rate', 'dso'.
        period: Year string e.g. '2024' to restrict the analysis window.
                Omit to analyse the current year.

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

    target_year = int(str(period).split("-")[0]) if period else date.today().year
    invoices = await _fetch_all_invoices_for_year(target_year)

    data_points: list[dict] = []

    if metric == "revenue":
        monthly: dict[str, float] = {}
        for inv in invoices:
            if inv.get("state") not in ("open", "partially_paid", "paid"):
                continue
            month = (inv.get("invoice_date") or "")[:7]
            if month:
                monthly[month] = monthly.get(month, 0.0) + float(inv.get("amount") or 0)
        data_points = [
            {"month": m, "value": round(v, 2)} for m, v in sorted(monthly.items())
        ]

    elif metric == "expenses":
        vouchers = await _fetch_all_vouchers_for_year(target_year)
        monthly = {}
        for v in vouchers:
            month = (v.get("date") or "")[:7]
            if month:
                monthly[month] = monthly.get(month, 0.0) + float(v.get("amount") or 0)
        data_points = [
            {"month": m, "value": round(v, 2)} for m, v in sorted(monthly.items())
        ]

    elif metric == "overdue_rate":
        today_str = date.today().isoformat()
        monthly_counts: dict[str, dict] = {}
        for inv in invoices:
            month = (inv.get("invoice_date") or "")[:7]
            if not month:
                continue
            if month not in monthly_counts:
                monthly_counts[month] = {"total": 0, "overdue": 0}
            monthly_counts[month]["total"] += 1
            if inv.get("state") == "open" and (inv.get("due_date") or "") < today_str:
                monthly_counts[month]["overdue"] += 1
        data_points = [
            {
                "month": m,
                "value": round(v["overdue"] / v["total"], 3) if v["total"] > 0 else 0.0,
            }
            for m, v in sorted(monthly_counts.items())
        ]

    else:  # dso
        monthly_dso: dict[str, list[float]] = {}
        for inv in invoices:
            if inv.get("state") != "paid":
                continue
            inv_date = inv.get("invoice_date") or ""
            due_date = inv.get("due_date") or ""
            month = inv_date[:7]
            if not month or len(inv_date) < 10 or len(due_date) < 10:
                continue
            try:
                dso = (date.fromisoformat(due_date) - date.fromisoformat(inv_date)).days
                if month not in monthly_dso:
                    monthly_dso[month] = []
                monthly_dso[month].append(float(dso))
            except ValueError:
                continue
        data_points = [
            {"month": m, "value": round(sum(vals) / len(vals), 1)}
            for m, vals in sorted(monthly_dso.items())
            if vals
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
