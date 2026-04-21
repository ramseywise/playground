"""Accounting domain tools for the Billy MCP server.

Covers Danish VAT (moms) reporting, audit readiness, period P&L summaries,
and accountant handoff document generation.

Danish VAT basics:
- Standard rate: 25% (moms)
- Reporting: quarterly for turnover < DKK 50M/year; bi-annually for < DKK 5M
- Output VAT: collected on sales (invoice.tax)
- Input VAT: paid on deductible purchases (expense.tax)
- Net VAT payable = output_vat - input_vat; negative = SKAT refund
- Due dates: Q1→1 Jun, Q2→1 Sep, Q3→1 Dec, Q4→1 Mar following year
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from app.db import get_conn


def get_vat_summary(quarter: int, year: int) -> dict:
    """Danish VAT (moms) summary for a reporting quarter.

    Output VAT: VAT collected on approved invoices (invoice.tax).
    Input VAT: VAT paid on deductible expenses (expense.tax).
    Net VAT payable = output_vat − input_vat.
    Positive = amount owed to SKAT; negative = refund position.

    Danish standard reporting is quarterly. Quarterly deadlines:
    Q1→1 Jun, Q2→1 Sep, Q3→1 Dec, Q4→1 Mar (following year).

    Args:
        quarter: 1, 2, 3, or 4.
        year: Calendar year, e.g. 2024.

    Returns:
        Dict with output_vat, input_vat, net_vat_payable, vat_rate_pct,
        currency, quarter, year, and status ("payable" or "refund").
    """
    if quarter not in (1, 2, 3, 4):
        return {"error": "quarter must be 1, 2, 3, or 4"}

    month_start = (quarter - 1) * 3 + 1
    month_end = quarter * 3
    year_str = str(year)

    with get_conn() as conn:
        output_vat = conn.execute(
            """SELECT COALESCE(SUM(tax), 0) FROM invoices
               WHERE state = 'approved'
                 AND strftime('%Y', entry_date) = ?
                 AND CAST(strftime('%m', entry_date) AS INTEGER) BETWEEN ? AND ?""",
            (year_str, month_start, month_end),
        ).fetchone()[0]

        input_vat = conn.execute(
            """SELECT COALESCE(SUM(tax), 0) FROM expenses
               WHERE strftime('%Y', date) = ?
                 AND CAST(strftime('%m', date) AS INTEGER) BETWEEN ? AND ?""",
            (year_str, month_start, month_end),
        ).fetchone()[0]

    net_vat_payable = round(output_vat - input_vat, 2)

    return {
        "year": year,
        "quarter": quarter,
        "output_vat": round(output_vat, 2),
        "input_vat": round(input_vat, 2),
        "net_vat_payable": net_vat_payable,
        "vat_rate_pct": 25.0,
        "currency": "DKK",
        "status": "payable" if net_vat_payable > 0 else "refund",
    }


def get_unreconciled_transactions(days_back: int = 30) -> dict:
    """Bank transactions not yet matched to any invoice.

    Unreconciled means matched_invoice_id IS NULL. Use
    match_transaction_to_invoice to reconcile each one.

    Args:
        days_back: Days to look back from today (1–365). Default 30.

    Returns:
        Dict with transactions list and total_unreconciled count.
    """
    days_back = max(1, min(days_back, 365))

    with get_conn() as conn:
        rows = conn.execute(
            """SELECT t.id, t.date, t.description, t.amount, t.type,
                      a.name AS account_name
               FROM bank_transactions t
               JOIN bank_accounts a ON a.id = t.account_id
               WHERE t.matched_invoice_id IS NULL
                 AND t.date >= date('now', ? || ' days')
               ORDER BY t.date DESC""",
            (f"-{days_back}",),
        ).fetchall()

    transactions = [
        {
            "id": r["id"],
            "date": r["date"],
            "description": r["description"],
            "amount": r["amount"],
            "type": r["type"],
            "account_name": r["account_name"],
        }
        for r in rows
    ]

    return {
        "days_back": days_back,
        "total_unreconciled": len(transactions),
        "transactions": transactions,
        "currency": "DKK",
    }


def get_audit_readiness_score() -> dict:
    """Compute an audit readiness score for the Billy account.

    Checks performed:
    1. Draft invoices (not yet approved) — medium severity
    2. Overdue invoices (approved, unpaid, past due date) — high severity
    3. Unreconciled bank transactions (no matched invoice) — medium severity
    4. Expenses missing description — low severity
    5. Invoice lines missing description — low severity

    Weighted deductions: high=20, medium=10, low=5 per failing check.
    Score = 100 minus deductions, clamped to 0.

    Returns:
        Dict with score (0–100), checks list, missing_docs, and recommendations.
    """
    today = date.today().isoformat()

    with get_conn() as conn:
        draft_count = conn.execute(
            "SELECT COUNT(*) FROM invoices WHERE state = 'draft'"
        ).fetchone()[0]

        overdue_count = conn.execute(
            "SELECT COUNT(*) FROM invoices WHERE state = 'approved' AND is_paid = 0 AND due_date < ?",
            (today,),
        ).fetchone()[0]

        unreconciled_count = conn.execute(
            "SELECT COUNT(*) FROM bank_transactions WHERE matched_invoice_id IS NULL"
        ).fetchone()[0]

        missing_expense_desc = conn.execute(
            "SELECT COUNT(*) FROM expenses WHERE description IS NULL OR description = ''"
        ).fetchone()[0]

        missing_line_desc = conn.execute(
            "SELECT COUNT(*) FROM invoice_lines WHERE description IS NULL OR description = ''"
        ).fetchone()[0]

    checks = [
        {
            "name": "Draft invoices",
            "status": "ok" if draft_count == 0 else "warning",
            "count": draft_count,
            "severity": "medium",
            "description": "Invoices not yet approved or sent to customers",
        },
        {
            "name": "Overdue invoices",
            "status": "ok" if overdue_count == 0 else "critical",
            "count": overdue_count,
            "severity": "high",
            "description": "Approved invoices past their due date with unpaid balance",
        },
        {
            "name": "Unreconciled bank transactions",
            "status": "ok" if unreconciled_count == 0 else "warning",
            "count": unreconciled_count,
            "severity": "medium",
            "description": "Bank transactions not matched to any invoice",
        },
        {
            "name": "Expenses missing description",
            "status": "ok" if missing_expense_desc == 0 else "info",
            "count": missing_expense_desc,
            "severity": "low",
            "description": "Expenses without description may fail documentation requirements",
        },
        {
            "name": "Invoice lines missing description",
            "status": "ok" if missing_line_desc == 0 else "info",
            "count": missing_line_desc,
            "severity": "low",
            "description": "Invoice line items without description",
        },
    ]

    _deductions = {"high": 20, "medium": 10, "low": 5}
    score = 100
    for c in checks:
        if c["status"] != "ok":
            score -= _deductions.get(c["severity"], 5)
    score = max(0, score)

    missing_docs: list[str] = []
    recommendations: list[str] = []

    if draft_count > 0:
        missing_docs.append(f"{draft_count} draft invoice(s) not yet approved")
        recommendations.append("Approve or void pending draft invoices before audit")
    if overdue_count > 0:
        missing_docs.append(f"{overdue_count} overdue invoice(s) awaiting payment")
        recommendations.append("Send payment reminders for overdue invoices")
    if unreconciled_count > 0:
        missing_docs.append(f"{unreconciled_count} unreconciled bank transaction(s)")
        recommendations.append(
            "Match bank transactions to invoices using match_transaction_to_invoice"
        )
    if missing_expense_desc > 0:
        missing_docs.append(f"{missing_expense_desc} expense(s) missing description")
        recommendations.append("Add descriptions to all expenses for documentation")
    if missing_line_desc > 0:
        missing_docs.append(f"{missing_line_desc} invoice line(s) missing description")
        recommendations.append("Add descriptions to invoice line items")

    if not missing_docs:
        recommendations.append("Account is audit-ready — no action items found.")

    return {
        "score": score,
        "max_score": 100,
        "checks": checks,
        "missing_docs": missing_docs,
        "recommendations": recommendations,
    }


def get_period_summary(year: int, quarter: Optional[int] = None) -> dict:
    """P&L summary for a year or quarter — suitable for accountant handoff.

    Revenue = approved invoice amounts (excl. VAT).
    Expenses = all expense amounts (excl. VAT).
    Profit = revenue − expenses.
    VAT position = output_vat − input_vat (positive = owed to SKAT).

    Args:
        year: Calendar year, e.g. 2024.
        quarter: Optional 1–4 to scope to a quarter. Omit for full year.

    Returns:
        Dict with year, quarter, revenue, expenses, profit, output_vat,
        input_vat, vat_position, invoice_count, expense_count, currency.
    """
    year_str = str(year)

    inv_conditions = ["state = 'approved'", "strftime('%Y', entry_date) = ?"]
    exp_conditions = ["strftime('%Y', date) = ?"]
    inv_params: list = [year_str]
    exp_params: list = [year_str]

    if quarter and quarter in (1, 2, 3, 4):
        month_start = (quarter - 1) * 3 + 1
        month_end = quarter * 3
        inv_conditions.append(
            "CAST(strftime('%m', entry_date) AS INTEGER) BETWEEN ? AND ?"
        )
        inv_params.extend([month_start, month_end])
        exp_conditions.append(
            "CAST(strftime('%m', date) AS INTEGER) BETWEEN ? AND ?"
        )
        exp_params.extend([month_start, month_end])

    inv_where = "WHERE " + " AND ".join(inv_conditions)
    exp_where = "WHERE " + " AND ".join(exp_conditions)

    with get_conn() as conn:
        inv_row = conn.execute(
            f"""SELECT COALESCE(SUM(amount), 0) AS revenue,
                       COALESCE(SUM(tax), 0) AS output_vat,
                       COUNT(*) AS invoice_count
                FROM invoices {inv_where}""",
            inv_params,
        ).fetchone()

        exp_row = conn.execute(
            f"""SELECT COALESCE(SUM(amount), 0) AS expenses,
                       COALESCE(SUM(tax), 0) AS input_vat,
                       COUNT(*) AS expense_count
                FROM expenses {exp_where}""",
            exp_params,
        ).fetchone()

    revenue = inv_row["revenue"]
    output_vat = inv_row["output_vat"]
    expenses = exp_row["expenses"]
    input_vat = exp_row["input_vat"]

    return {
        "year": year,
        "quarter": quarter,
        "revenue": round(revenue, 2),
        "expenses": round(expenses, 2),
        "profit": round(revenue - expenses, 2),
        "output_vat": round(output_vat, 2),
        "input_vat": round(input_vat, 2),
        "vat_position": round(output_vat - input_vat, 2),
        "invoice_count": inv_row["invoice_count"],
        "expense_count": exp_row["expense_count"],
        "currency": "DKK",
    }


def generate_handoff_doc(year: int, quarter: Optional[int] = None) -> dict:
    """Generate a markdown P&L handoff document for accountant review.

    Combines get_period_summary and get_audit_readiness_score into a
    bilingual (Danish headings) markdown document with a VAT table,
    P&L summary, and documentation status section.

    Args:
        year: Calendar year, e.g. 2024.
        quarter: Optional 1–4. Omit for full-year summary.

    Returns:
        Dict with period_label, markdown_summary, and missing_items list.
    """
    period = get_period_summary(year=year, quarter=quarter)
    audit = get_audit_readiness_score()

    period_label = f"Q{quarter} {year}" if quarter else str(year)

    lines = [
        f"# Regnskabsopsummering — {period_label}",
        "",
        "## Resultatopgørelse",
        "",
        "| Post | Beløb (DKK) |",
        "|------|------------|",
        f"| Omsætning (ekskl. moms) | {period['revenue']:,.2f} |",
        f"| Omkostninger (ekskl. moms) | {period['expenses']:,.2f} |",
        f"| **Resultat** | **{period['profit']:,.2f}** |",
        "",
        "## Momsopgørelse",
        "",
        "| Post | Beløb (DKK) |",
        "|------|------------|",
        f"| Salgsmoms (output VAT) | {period['output_vat']:,.2f} |",
        f"| Købsmoms (input VAT) | {period['input_vat']:,.2f} |",
        f"| **Skyldig moms** | **{period['vat_position']:,.2f}** |",
        "",
    ]

    if period["vat_position"] >= 0:
        lines.append("> Positiv balance = skyldig moms til SKAT")
    else:
        lines.append("> Negativ balance = momsgodtgørelse fra SKAT")

    lines += [
        "",
        "## Dokumentationsstatus",
        "",
        f"Audit-score: **{audit['score']}/100**",
        "",
    ]

    if audit["missing_docs"]:
        lines.append("**Manglende/ufuldstændige poster:**")
        lines.append("")
        for item in audit["missing_docs"]:
            lines.append(f"- {item}")
        lines.append("")
    else:
        lines.append("Ingen manglende poster — klar til revision.")
        lines.append("")

    lines += [
        "## Nøgletal",
        "",
        f"- Antal godkendte fakturaer: {period['invoice_count']}",
        f"- Antal udgifter registreret: {period['expense_count']}",
        "",
        "---",
        "_Genereret af Billy regnskabsassistent_",
    ]

    return {
        "period_label": period_label,
        "markdown_summary": "\n".join(lines),
        "missing_items": audit["missing_docs"],
    }
