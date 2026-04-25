"""
Ingest sevdesk single-engagement tickets into Billy VA eval fixtures.

Covers all 7 CES rating levels with stratified per-level sampling.
Deterministic regex pass only (email, phone, IBAN, postal, salutation names).
After running, follow .claude/skills/gdpr-scrub/SKILL.md for the LLM review
pass before committing the output file.

Usage:
    cd va-langgraph
    uv run python eval/ingest/sevdesk_ingest.py
    uv run python eval/ingest/sevdesk_ingest.py --n 280 --output path/to/out.json
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
# CES → test signal mapping (each level is a distinct friction proxy)
# ---------------------------------------------------------------------------

# CES 1 = low effort (easy win) → gold standard capability
# CES 7 = high effort (frustrated) → regression / failure mode
CES_TEST_TYPE: dict[int, str] = {
    1: "capability",       # VA handled it perfectly, zero friction
    2: "near_win",         # Easy path with minor friction
    3: "friction_low",     # Friction starting to emerge
    4: "baseline",         # Neutral — no clear positive or negative signal
    5: "friction_high",    # Customer showing frustration
    6: "pre_escalation",   # High escalation risk, customer close to leaving
    7: "regression",       # Failure mode — maximal frustration
}

CES_DIFFICULTY: dict[int, str] = {
    1: "easy",
    2: "easy",
    3: "medium",
    4: "medium",
    5: "medium",
    6: "hard",
    7: "hard",
}

# ---------------------------------------------------------------------------
# Escalation signal
# All TE- (technical/engineering) categories require human escalation —
# the VA cannot fix platform bugs or incidents.
# Some SE- (billing/account) categories require account access or approval.
# ---------------------------------------------------------------------------

_SE_ESCALATION_CATEGORIES: set[str] = {
    "SE - Account gesperrt",             # account lock — needs ops access
    "SE - Rechnung und Zahlungsdetails", # billing details — needs account verification
    "SE - Rechnungsüberweisung",         # wire transfer dispute — needs finance team
    "SE - Inkasso / PairFinance",        # debt collection referral
    "SE - Auskunftsersuchen",            # GDPR data request — legal obligation
}


def _is_escalation(category: str) -> bool:
    """True when this ticket type should route to a human, not the VA."""
    return category.startswith("TE - ") or category in _SE_ESCALATION_CATEGORIES


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
# Salutation + first name: "Hallo Nicole," / "Guten Tag Max" / "Sehr geehrter Herr Müller"
_SALUTE_RE = re.compile(
    r"(Hallo|Guten\s+Tag|Liebe[r]?|Dear|Sehr geehrte[r]?(?:\s+(?:Herr|Frau))?)\s+([A-ZÄÖÜ][a-zäöüß]{1,})",
    re.IGNORECASE,
)
# Ticket/account refs: 7+ digit standalone, OR 6+ digit prefixed by account keyword
_TICKET_REF_RE = re.compile(
    r"\b(Ticket|Ticketnummer|Ticketname)\s*[:#]?\s*\d{7,}"
    r"|\b(Kundennummer|Kunden-?Nr\.?|Account|Konto)\s*[:#]?\s*\d{6,}",
    re.IGNORECASE,
)
# Business registration / tax IDs (AT/DE UID, Firmennummer)
_BIZ_ID_RE = re.compile(r"\b(UID|ATU|USt-?IdNr\.?|FN)\s*[:\.]?\s*[A-Z0-9]{6,}", re.IGNORECASE)


def _scrub(text: str) -> str:
    text = _ANGLE_URL_RE.sub("", text)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _IBAN_RE.sub("[IBAN]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _POSTAL_RE.sub("[LOCATION]", text)
    text = _SALUTE_RE.sub(lambda m: f"{m.group(1)} [NAME]", text)
    text = _TICKET_REF_RE.sub(lambda m: f"{m.group(1) or m.group(2)} [REF]", text)
    text = _BIZ_ID_RE.sub(lambda m: f"{m.group(1)} [BIZ-ID]", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Reply chain stripping
# ---------------------------------------------------------------------------

# Catches both German email-client formats:
#   "Am 2. Jan. 2025 schrieb Max:" (Outlook/Thunderbird style)
#   "Max Mustermann schrieb am Do. 2. Jan.:" (forwarded message style)
_CHAIN_RE = re.compile(
    r"\n?(Am\s.{5,120}schrieb"
    r"|.{0,80}schrieb am\s+\w"
    r"|Von:\s|From:\s"
    r"|-----+\s*Original"
    r"|Gesendet:\s"
    r"|On\s.{5,120}wrote:).*",
    re.IGNORECASE | re.DOTALL,
)

# Angle-bracket URLs left by email clients in forwarded/quoted blocks
# e.g. <https://cBMph04.na1.hubspotlinks.com/...>
_ANGLE_URL_RE = re.compile(r"<https?://\S+?>", re.IGNORECASE)


def _strip_chain(text: str) -> str:
    m = _CHAIN_RE.search(text)
    return text[: m.start()].strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Agent signature stripping from ENGAGEMENT_MESSAGE
# ---------------------------------------------------------------------------

# Matches closing lines + everything after, including "Dein/e [Name]" agent sign-off
_SIG_RE = re.compile(
    r"\n{1,2}(Viele Grüße|Mit freundlichen Grüßen|Best regards|Kind regards|Freundliche Grüße"
    r"|Dein[e]?\s+[A-ZÄÖÜ][a-zäöüß]+)[^\n]*(?:\n.*)?",
    re.IGNORECASE | re.DOTALL,
)


def _strip_signature(text: str) -> str:
    m = _SIG_RE.search(text)
    return text[: m.start()].strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Sevdesk agent response boilerplate — every reply appends:
#   "Deine Ticketnummer:\n{ID}\n\nTicketbeschreibung:\n{full original message}"
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
# Stratified sampling: per-CES-level quota, then balanced by category within
# ---------------------------------------------------------------------------

def _balance_by_category(rows: list[dict], quota: int) -> list[dict]:
    """Pick up to `quota` rows, balanced across TICAT categories."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = r["TICAT_AGENT_LABELLED_CATEGORY"] or "SU - Sonstiges"
        buckets[key].append(r)

    if not buckets:
        return []

    per_bucket = max(1, quota // len(buckets))
    selected: list[dict] = []
    overflow: list[dict] = []

    for bucket in buckets.values():
        selected.extend(bucket[:per_bucket])
        overflow.extend(bucket[per_bucket:])

    needed = quota - len(selected)
    if needed > 0:
        selected.extend(overflow[:needed])

    return selected[:quota]


def _sample_stratified(rows: list[dict], total: int) -> list[dict]:
    """Guarantee ~equal representation across all CES levels present in rows,
    then balance by category within each level."""
    by_ces: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ces[r["CES_RATING_LAST"]].append(r)

    ces_levels = sorted(by_ces.keys(), key=int)
    quota_per_level = max(1, total // len(ces_levels))

    selected: list[dict] = []
    leftovers: list[dict] = []

    for level in ces_levels:
        level_rows = by_ces[level]
        level_pick = _balance_by_category(level_rows, quota_per_level)
        selected.extend(level_pick)
        picked_ids = {id(r) for r in level_pick}
        leftovers.extend(r for r in level_rows if id(r) not in picked_ids)

    needed = total - len(selected)
    if needed > 0:
        selected.extend(leftovers[:needed])

    return selected[:total]


# ---------------------------------------------------------------------------
# Fixture assembly
# ---------------------------------------------------------------------------

def _build_fixture(row: dict, seq: int) -> dict:
    category = row["TICAT_AGENT_LABELLED_CATEGORY"] or "SU - Sonstiges"
    intent = CATEGORY_INTENT.get(category, _DEFAULT_INTENT)
    ces = int(row["CES_RATING_LAST"])
    escalation = _is_escalation(category)

    raw_query = _strip_chain(row["CONTENT"] or "")
    raw_answer = _strip_ticket_boilerplate(
        _strip_chain(_strip_signature(row["ENGAGEMENT_MESSAGE"] or ""))
    )

    query = _scrub(raw_query)
    answer = _scrub(raw_answer)

    tags = ["sevdesk", "real-ticket", f"ces-{ces}"]
    if escalation:
        tags.append("escalation-candidate")

    return {
        "id": f"sev-{seq:03d}",
        "query": query,
        "expected_answer": answer,
        "expected_intent": intent,
        "category": intent,
        "ces_rating": ces,
        "test_type": CES_TEST_TYPE.get(ces, "baseline"),
        "difficulty": CES_DIFFICULTY.get(ces, "medium"),
        "escalation_signal": escalation,
        "source": "sevdesk_raw",
        "language": "de",
        "source_category": category,
        "tags": tags,
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

    # "Predicted Category" prefix = TICAT auto-labeller reasoning leaked into CONTENT
    _CLASSIFICATION_NOTE_RE = re.compile(r"(Predicted Category|I've chosen\s+')", re.IGNORECASE)

    eligible = [
        r for r in all_rows
        if r["CES_RATING_LAST"].strip()
        and (r["TICAT_AGENT_LABELLED_CATEGORY"] or "") not in SKIP_CATEGORIES
        and (r["CONTENT"] or "").strip()
        and (r["ENGAGEMENT_MESSAGE"] or "").strip()
        and not _CLASSIFICATION_NOTE_RE.search(r["CONTENT"] or "")
    ]

    print(f"Total rows:     {len(all_rows):>6}", file=sys.stderr)
    print(f"CES-rated:      {sum(1 for r in all_rows if r['CES_RATING_LAST'].strip()):>6}", file=sys.stderr)
    print(f"Eligible:       {len(eligible):>6}", file=sys.stderr)

    by_ces_raw = Counter(r["CES_RATING_LAST"] for r in eligible)
    print(f"By CES (eligible): {dict(sorted(by_ces_raw.items(), key=lambda x: int(x[0])))}", file=sys.stderr)

    sampled = _sample_stratified(eligible, n)
    print(f"Sampled:        {len(sampled):>6}", file=sys.stderr)

    raw_fixtures = [_build_fixture(r, i + 1) for i, r in enumerate(sampled)]
    # Drop anything that cleaned down to nothing — usually an all-chain forwarded message
    fixtures = [f for f in raw_fixtures if f["query"].strip() and f["expected_answer"].strip()]
    n_dropped = len(raw_fixtures) - len(fixtures)
    if n_dropped:
        print(f"Dropped (empty after clean): {n_dropped}", file=sys.stderr)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2))
    print(f"\nWritten → {output}", file=sys.stderr)

    by_ces = Counter(f["ces_rating"] for f in fixtures)
    by_intent = Counter(f["expected_intent"] for f in fixtures)
    by_type = Counter(f["test_type"] for f in fixtures)
    n_escalation = sum(1 for f in fixtures if f["escalation_signal"])

    print(f"\nBy CES rating:  {dict(sorted(by_ces.items()))}", file=sys.stderr)
    print(f"By test_type:   {dict(sorted(by_type.items()))}", file=sys.stderr)
    print(f"By intent:      {dict(sorted(by_intent.items()))}", file=sys.stderr)
    print(f"Escalation:     {n_escalation} of {len(fixtures)} ({100*n_escalation//len(fixtures)}%)", file=sys.stderr)
    print(f"\nNext step: follow .claude/skills/gdpr-scrub/SKILL.md — LLM review pass", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_CSV, help="Path to single_engagement_tickets.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON, help="Output fixture JSON path")
    parser.add_argument("--n", type=int, default=280, help="Number of fixtures to sample (default: 280)")
    args = parser.parse_args()
    main(args.source, args.output, args.n)
