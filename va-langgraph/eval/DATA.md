# Eval Dataset: Clara Real Tickets

Source: `help-support-rag-agent/data/single_engagement_tickets.csv`
Ingest script: `eval/ingest/clara_ingest.py`
Fixture file: `tests/evalsuite/fixtures/clara_tickets.json`

To regenerate: `make va-eval-ingest` → `make va-eval-review` → `make va-eval-pii-check`

---

## Current Batch (v1 — clara_raw)

| Metric | Value |
|---|---|
| Total source rows | 52,215 |
| CES-rated rows | 2,364 (4.5%) |
| Eligible after filter | 2,283 |
| Sampled fixtures | **278** |
| Language | German (de) |
| Source label | `clara_raw` |

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
| SE - Tarifwechsel | 78 | (skipped — CRM meta) |
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
| German/Austrian phone numbers (+49/+43/0 prefix) | `[PHONE]` |
| German IBANs (DE + 20 digits) | `[IBAN]` |
| Postal code + city (4-digit AT + 5-digit DE) | `[LOCATION]` |
| Salutation + first name | `[NAME]` |
| Herr/Frau + name in body text | `[NAME]` |
| Street addresses (Musterstraße 12, Hauptstr. 5a) | `[ADDRESS]` |
| Ticket/account reference (7+ digits) | `[REF]` |
| Business registration IDs (UID, ATU, FN, HRA, HRB) | `[BIZ-ID]` |
| Angle-bracket URLs from email clients | *(stripped)* |
| Email reply chains / forwarded messages | *(stripped)* |
| Agent signature blocks | *(stripped)* |
| Clara ticket boilerplate footer | *(stripped)* |

**Also filtered at ingestion:**
- TICAT classification notes leaked into `CONTENT` field
- Fixtures where query or answer cleaned to empty (forwarded-only messages)

### Pass 2: LLM Review (manual, run after ingest)

Two review passes were run on this dataset.

**v1 review — full batch (278 fixtures):** 195 findings, all high confidence.

| Type | Count | Notes |
|---|---|---|
| name | 148 | Full names in body text not following Herr/Frau or a salutation |
| address | 22 | Unusual formats: Bahnhofstr., "Am Nelkenberg 13" |
| company | 15 | Personal company names: "Müller GmbH", "Fa. Erich Wutzig" |
| account_id | 6 | Short IDs (6 digits), B2B refs, court case numbers |
| other | 3 | Personal URLs (www.erikbender.de), tracking tokens |
| phone | 1 | Austrian `T +43` format missed by phone regex |

**Regex improvements made after v1 review** (improvements to `clara_ingest.py`):
- `_TITLED_NAME_RE`: `Herr/Frau + Name` in body text → `Herr/Frau [NAME]`
- `_ADDRESS_RE`: Straße/Weg/Allee/Platz + house number → `[ADDRESS]`
- `_PHONE_RE`: extended to cover Austrian `+43` and `T +43` prefix
- `_POSTAL_RE`: extended to 4-digit AT postal codes
- `_HRX_RE`: `HRA/HRB \d{4,}` commercial register → `HRB [BIZ-ID]`

**v2 review — after applying all 195 findings + regex improvements:** 22 findings.

| Type | Count | Notes |
|---|---|---|
| name | 7 | ALL-CAPS hyphenated names, Slavic name suffixes after `[NAME]` |
| url | 6 | Personal websites (www.engelhardt-atelier.de), screen recordings |
| account_id | 7 | RE-prefix invoice numbers, short 5-6 digit IDs |
| other | 2 | CID image URLs with embedded name |

**False positives identified in v2 review (5 findings, not applied):**
- `Hallo [NAME] [NAME]` — LLM flagged first+last name already correctly replaced
- `liebe [NAME]\n[NAME]` — same: greeting + closing signature, both replaced
- `[NAME]\n\n[NAME]` — two adjacent placeholders in signature area
- GitHub public repo URLs — not personal PII (public repos shared by support agent)

**All real findings applied.** Final state: PII check passes on all 278 fixtures.

Run: `make va-eval-review` — uses Gemini 2.5 Flash, outputs structured findings JSON.
Save to file: `cd va-langgraph && uv run python eval/ingest/gdpr_review.py --findings eval/ingest/gdpr_findings.json`

---

## Friction Analysis Notes

CES ratings as friction signal proxies — planned analysis:
- **CES 1 vs 7 gap by category:** which topics consistently frustrate vs delight?
- **Escalation rate by CES:** do CES 6-7 tickets cluster in escalation categories?
- **Intent × CES heatmap:** where does the VA have structural coverage vs gaps?

These slices require the fixture set to be committed and the VA responses to be
collected — use `clara_capability_tasks` and `clara_regression_tasks` fixtures
in `tests/evalsuite/conftest.py`.

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| v1 — clara_raw | in progress | 278 fixtures, German, Clara (sevdesk) refs intact |
| v2 — clara_adapted | planned | Replace sevdesk→Billy refs, update source field |
| v3 — multi-turn | planned | Conversation chains, not single QA pairs |
| Friction analysis | planned | CES × category × escalation heatmap |
| RAG coverage map | planned | Which topics are in the knowledge base vs not |
