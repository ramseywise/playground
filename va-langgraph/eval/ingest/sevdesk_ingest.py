"""
Ingest sevdesk single-engagement tickets into Billy VA eval fixtures.

Deterministic regex pass only (email, phone, IBAN, postal, salutation names).
After running, follow .claude/skills/gdpr-scrub/SKILL.md for the LLM review
pass before committing the output file.

Usage:
    cd va-langgraph
    uv run python eval/ingest/sevdesk_ingest.py
    uv run python eval/ingest/sevdesk_ingest.py --n 100 --output path/to/out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[3]  # playground/
SOURCE_CSV = _REPO_ROOT.parent / "help-support-rag-agent" / "data" / "single_engagement_tickets.csv"
OUTPUT_JSON = _REPO_ROOT / "va-langgraph" / "tests" / "evalsuite" / "fixtures" / "sevdesk_tickets.json"

# ---------------------------------------------------------------------------
# Category rules
# ---------------------------------------------------------------------------

# Sevdesk-meta categories with no Billy equivalent — skip entirely
SKIP_CATEGORIES: set[str] = {
    "SE - Vertrag kündigen",
    "SE - Upgrade/Downgrade",
    "SE - Testzugang / Erstanmeldung",
    "SE - Partnerprogramm",
}

# Sevdesk TICAT label → Billy VA intent
CATEGORY_INTENT: dict[str, str] = {
    "SU - Rechnungen & Dokumente erstellen und löschen": "invoice",
    "SU - Rechnungen & Belege verbuchen": "accounting",
    "SU - Auswertungen & Umsatzsteuer": "insights",
    "SU - Angebote & Teil-/Abschlagsrechnungen": "quote",
    "SU - Dokument-Export (PDF, CSV, DATEV)": "accounting",
    "SU - Kundenverwaltung": "customer",
    "SU - Bankimport": "banking",
    "SU - Datev-Export": "accounting",
    "SU - Login & Passwort-Probleme": "support",
    "SU - Sonstiges": "support",
    "SE - Rechnung und Zahlungsdetails": "invoice",
    "SE - Rechnungsüberweisung": "banking",
    "SE - Kulanz / Rabattanfrage": "support",
    "SE - Sonstiges": "support",
    "TE - Transaktionen abrufen/importieren": "banking",
}
_DEFAULT_INTENT = "support"

# ---------------------------------------------------------------------------
# PII scrubbing — regex pass
# ---------------------------------------------------------------------------

# Full email with TLD, plus bare @domain patterns (catches truncated/no-TLD cases)
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    r"|[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]{3,}",
    re.IGNORECASE,
)
_IBAN_RE = re.compile(r"\bDE\d{20}\b")
# German phone: +49 or leading 0, followed by 7+ digits (allows spaces/dashes)
_PHONE_RE = re.compile(r"(\+49|0\d{1,4})[\s\-\./]?\d[\d\s\-\.\/]{5,}\d")
# German 5-digit postal code followed by city name
_POSTAL_RE = re.compile(r"\b\d{5}\s+[A-ZÄÖÜ][a-zäöüß\-]+")
# Salutation + first name: "Hallo Nicole," / "Lieber Max!" / "Sehr geehrter Herr Müller"
_SALUTE_RE = re.compile(
    r"(Hallo|Guten\s+Tag|Liebe[r]?|Dear|Sehr geehrte[r]?(?:\s+(?:Herr|Frau))?)\s+([A-ZÄÖÜ][a-zäöüß]{1,})",
    re.IGNORECASE,
)


_TICKET_REF_RE = re.compile(r"\b(Ticket|Ticketnummer|Ticketname)\s*[:#]?\s*\d{7,}", re.IGNORECASE)
# Business registration / tax IDs (AT/DE UID, Firmennummer)
_BIZ_ID_RE = re.compile(r"\b(UID|ATU|USt-?IdNr\.?|FN)\s*[:\.]?\s*[A-Z0-9]{6,}", re.IGNORECASE)


def _scrub(text: str) -> str:
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _IBAN_RE.sub("[IBAN]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _POSTAL_RE.sub("[LOCATION]", text)
    text = _SALUTE_RE.sub(lambda m: f"{m.group(1)} [NAME]", text)
    text = _TICKET_REF_RE.sub(lambda m: f"{m.group(1)} [REF]", text)
    text = _BIZ_ID_RE.sub(lambda m: f"{m.group(1)} [BIZ-ID]", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Reply chain stripping
# ---------------------------------------------------------------------------

_CHAIN_RE = re.compile(
    r"\n?(Am\s.{5,120}schrieb|Von:\s|From:\s|-----+\s*Original|Gesendet:\s|On\s.{5,120}wrote:).*",
    re.IGNORECASE | re.DOTALL,
)


def _strip_chain(text: str) -> str:
    m = _CHAIN_RE.search(text)
    return text[: m.start()].strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Agent signature stripping from ENGAGEMENT_MESSAGE
# Signatures appear after the substantive reply, separated by blank line + name
# ---------------------------------------------------------------------------

_SIG_RE = re.compile(
    r"\n{1,2}(Viele Grüße|Mit freundlichen Grüßen|Best regards|Kind regards|Freundliche Grüße)[^\n]*\n.*",
    re.IGNORECASE | re.DOTALL,
)


def _strip_signature(text: str) -> str:
    m = _SIG_RE.search(text)
    return text[: m.start()].strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Sevdesk agent response boilerplate — every reply contains a footer block:
#   "Deine Ticketnummer:\n{ID}​\n\nTicketbeschreibung:\n{full original message}"
# This repeats the original ticket and leaks the ticket ID. Strip it.
# ---------------------------------------------------------------------------

_TICKET_BOILERPLATE_RE = re.compile(
    r"\n?\s*(Deine\s+)?Ticketn(ummer|ame)\s*:\s*\n.*",
    re.IGNORECASE | re.DOTALL,
)


def _strip_ticket_boilerplate(text: str) -> str:
    m = _TICKET_BOILERPLATE_RE.search(text)
    return text[: m.start()].strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Balanced sampling across (ces_rating × category) buckets
# ---------------------------------------------------------------------------

def _sample_balanced(rows: list[dict], total: int) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["CES_RATING_LAST"], r["TICAT_AGENT_LABELLED_CATEGORY"] or "SU - Sonstiges")
        buckets[key].append(r)

    n_buckets = len(buckets)
    quota = max(1, total // n_buckets)

    selected: list[dict] = []
    overflow: list[dict] = []

    for bucket in buckets.values():
        selected.extend(bucket[:quota])
        overflow.extend(bucket[quota:])

    # Fill remaining slots from overflow (largest buckets first)
    needed = total - len(selected)
    if needed > 0:
        selected.extend(overflow[:needed])

    return selected[:total]


# ---------------------------------------------------------------------------
# Fixture assembly
# ---------------------------------------------------------------------------

def _build_fixture(row: dict, seq: int) -> dict:
    category = row["TICAT_AGENT_LABELLED_CATEGORY"] or "SU - Sonstiges"
    intent = CATEGORY_INTENT.get(category, _DEFAULT_INTENT)
    ces = int(row["CES_RATING_LAST"])

    raw_query = _strip_chain(row["CONTENT"] or "")
    raw_answer = _strip_ticket_boilerplate(
        _strip_chain(_strip_signature(row["ENGAGEMENT_MESSAGE"] or ""))
    )

    query = _scrub(raw_query)
    answer = _scrub(raw_answer)

    return {
        "id": f"sev-{seq:03d}",
        "query": query,
        "expected_answer": answer,
        "expected_intent": intent,
        "category": intent,
        "ces_rating": ces,
        # CES 1 = low effort (easy win) → capability; CES 7 = high effort (frustrated) → regression
        "difficulty": "easy" if ces == 1 else "hard",
        "test_type": "capability" if ces == 1 else "regression",
        "source": "sevdesk_raw",
        "language": "de",
        "source_category": category,
        "tags": ["sevdesk", "real-ticket", f"ces-{ces}"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(source: Path, output: Path, n: int) -> None:
    if not source.exists():
        print(f"ERROR: source not found: {source}", file=sys.stderr)
        print("Expected at: help-support-rag-agent/data/single_engagement_tickets.csv", file=sys.stderr)
        sys.exit(1)

    with source.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    eligible = [
        r for r in all_rows
        if r["CES_RATING_LAST"] in ("1", "7")
        and (r["TICAT_AGENT_LABELLED_CATEGORY"] or "") not in SKIP_CATEGORIES
        and (r["CONTENT"] or "").strip()
        and (r["ENGAGEMENT_MESSAGE"] or "").strip()
    ]

    print(f"Total rows:     {len(all_rows):>6}", file=sys.stderr)
    print(f"CES-rated:      {sum(1 for r in all_rows if r['CES_RATING_LAST']):>6}", file=sys.stderr)
    print(f"Eligible (1/7): {len(eligible):>6}", file=sys.stderr)

    sampled = _sample_balanced(eligible, n)
    print(f"Sampled:        {len(sampled):>6}", file=sys.stderr)

    fixtures = [_build_fixture(r, i + 1) for i, r in enumerate(sampled)]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2))
    print(f"\nWritten → {output}", file=sys.stderr)

    by_ces = Counter(f["ces_rating"] for f in fixtures)
    by_intent = Counter(f["expected_intent"] for f in fixtures)
    print(f"\nBy CES rating: {dict(sorted(by_ces.items()))}", file=sys.stderr)
    print(f"By intent:     {dict(sorted(by_intent.items()))}", file=sys.stderr)
    print(f"\nNext step: follow .claude/skills/gdpr-scrub/SKILL.md — LLM review pass", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_CSV, help="Path to single_engagement_tickets.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON, help="Output fixture JSON path")
    parser.add_argument("--n", type=int, default=50, help="Number of fixtures to sample (default: 50)")
    args = parser.parse_args()
    main(args.source, args.output, args.n)
