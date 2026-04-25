# Eval Dataset: Sevdesk Real Tickets

Source: `help-support-rag-agent/data/single_engagement_tickets.csv`  
Ingest script: `eval/ingest/sevdesk_ingest.py`  
Fixture file: `tests/evalsuite/fixtures/sevdesk_tickets.json`

To regenerate: `make va-eval-ingest` → `make va-eval-review` → `make va-eval-pii-check`

---

## Current Batch (v1 — sevdesk_raw)

| Metric | Value |
|---|---|
| Total source rows | 52,215 |
| CES-rated rows | 2,364 (4.5%) |
| Eligible after filter | 2,283 |
| Sampled fixtures | **278** |
| Language | German (de) |
| Source label | `sevdesk_raw` |

### Distribution

**By CES rating** (40 per level, stratified):

| CES | test_type | Signal | Count |
|---|---|---|---|
| 1 | `capability` | Gold standard — VA handled with zero friction | 40 |
| 2 | `near_win` | Easy path with minor friction | 39 |
| 3 | `friction_low` | Friction starting to emerge | 40 |
| 4 | `baseline` | Neutral — no clear signal | 40 |
| 5 | `friction_high` | Customer showing frustration | 39 |
| 6 | `pre_escalation` | High escalation risk | 40 |
| 7 | `regression` | Failure mode / maximal frustration | 40 |

**By intent:**

| Intent | Count | Notes |
|---|---|---|
| support | 188 | Catch-all; includes misc SU/SE tickets |
| invoice | 32 | Rechnungen erstellen, Zahlungsdetails |
| accounting | 21 | Belege verbuchen, DATEV, Umsatzsteuer |
| insights | 18 | Auswertungen & USt |
| banking | 12 | Bankimport, Rechnungsüberweisung |
| quote | 7 | Angebote & Abschlagsrechnungen |

**Escalation signal:** 66 of 278 (24%) tagged `escalation-candidate`

---

## Escalation Signal Analysis

~1 in 4 tickets in this dataset are types the VA should not try to answer —
they require human access, technical investigation, or legal authority.

**Always-escalation categories (TE- prefix — platform/engineering issues):**
- `TE - Fehlermeldung` — 73% CES 7 skew
- `TE - Server Fehlermeldung` — 75% CES 7 skew
- `TE - Incident Pipeline` — 75% CES 7 skew
- `TE - FinAPI Berechtigungs Fehler`, `TE - Neo-Banken`, `TE - API`, etc.

**Billing/account escalations (SE- high-frustration):**
- `SE - Rechnung und Zahlungsdetails` — 67% CES 7, needs account verification
- `SE - Rechnungsüberweisung` — 62% CES 7, wire transfer disputes
- `SE - Account gesperrt` — 58% CES 7, locked account → ops access required
- `SE - Inkasso / PairFinance` — debt collection referral
- `SE - Auskunftsersuchen` — GDPR data requests, legal obligation

**Implication for RAG coverage:** the `SU -` category prefix (Support / product how-to)
is the space where RAG can add value. The remaining 24% are structural escalations
regardless of answer quality.

---

## Topic Coverage

58 unique TICAT categories in eligible tickets. Top 10 by volume:

| Category | Total | Billy intent |
|---|---|---|
| SU - Rechnungen & Dokumente erstellen und löschen | 461 | invoice |
| SU - Sonstiges | 220 | support |
| SU - Rechnungen & Belege verbuchen | 202 | accounting |
| SU - Auswertungen & Umsatzsteuer | 153 | insights |
| SE - Rechnung und Zahlungsdetails | 107 | invoice |
| SU - Angebote & Teil-/Abschlagsrechnungen | 97 | quote |
| SE - Rechnungsüberweisung | 81 | banking |
| SE - Tarifwechsel | 78 | (skipped — sevdesk meta) |
| SU - Login & Passwort-Probleme | 73 | support |
| TE - Transaktionen abrufen/importieren | 65 | banking |

With 278 fixtures across 58 categories, coverage is ~4.8 per category on average.
Long-tail categories (< 10 eligible tickets) may appear 0-1 times in the fixture set.

---

## PII Scrubbing — Two-Pass Approach

### Pass 1: Regex (automated, runs in ingest script)

| Pattern | Placeholder |
|---|---|
| Email addresses | `[EMAIL]` |
| German phone numbers (+49 / 0 prefix) | `[PHONE]` |
| German IBANs (DE + 20 digits) | `[IBAN]` |
| Postal code + city | `[LOCATION]` |
| Salutation + first name | `[NAME]` |
| Ticket/account reference (7+ digits) | `[REF]` |
| Business registration IDs (UID, ATU, FN) | `[BIZ-ID]` |
| Angle-bracket URLs from email clients | *(stripped)* |
| Email reply chains / forwarded messages | *(stripped)* |
| Agent signature blocks | *(stripped)* |
| Sevdesk ticket boilerplate footer | *(stripped)* |

**Also filtered at ingestion:**
- TICAT classification notes leaked into `CONTENT` field
- Fixtures where query or answer cleaned to empty (forwarded-only messages)

### Pass 2: LLM Review (manual, run after ingest)

Calibration run on 14 fixtures found **9 findings** — regex misses ~64% of fixtures
have at least one residual PII item needing manual remediation.

**Common gaps the regex misses:**

| Gap | Example | Fix |
|---|---|---|
| Full names in body text | "Jürgen Mann" referenced mid-sentence | `[NAME]` |
| Street address without postal code | "Hallstädter Weg 1" | `[ADDRESS]` |
| Short account IDs with keyword | "Kundennummer 368488" (6 digits) | `[REF]` |
| Agent first name in closing | "Dein Felix" after signature stripping | `[NAME]` |

Run: `make va-eval-review` — uses Gemini 2.5 Flash, outputs structured findings JSON.

---

## Friction Analysis Notes

CES ratings as friction signal proxies — planned analysis:
- **CES 1 vs 7 gap by category:** which topics consistently frustrate vs delight?
- **Escalation rate by CES:** do CES 6-7 tickets cluster in escalation categories?
- **Intent × CES heatmap:** where does the VA have structural coverage vs gaps?

These slices require the fixture set to be committed and the VA responses to be
collected — use `sevdesk_capability_tasks` and `sevdesk_regression_tasks` fixtures
in `tests/evalsuite/conftest.py`.

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| v1 — sevdesk_raw | in progress | 278 fixtures, German, sevdesk refs intact |
| v2 — sevdesk_adapted | planned | Replace sevdesk→Billy refs, update source field |
| v3 — multi-turn | planned | Conversation chains, not single QA pairs |
| Friction analysis | planned | CES × category × escalation heatmap |
| RAG coverage map | planned | Which topics are in the knowledge base vs not |
