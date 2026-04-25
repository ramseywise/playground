---
name: gdpr-scrub
description: GDPR/PII compliance process for ingesting real customer service tickets into eval fixtures. Covers German-language sevdesk tickets: regex pre-pass, LLM review pass, placeholder convention, and commit checklist. Invoke before committing any fixture derived from real user data.
---

# GDPR Scrub: Eval Data Ingestion

Use when ingesting real customer conversation data into the eval fixture pipeline. The goal is to remove all personal identifiers while preserving the semantic content needed to test the agent.

---

## Scope

Currently applies to:
- `help-support-rag-agent/data/single_engagement_tickets.csv` → Billy VA eval fixtures
- Any future import of real CS ticket data

---

## Step 1: Run the Ingestion Script (deterministic regex pass)

```bash
cd va-langgraph
uv run python eval/ingest/sevdesk_ingest.py \
  --output tests/evalsuite/fixtures/sevdesk_tickets.json \
  --n 50
```

The script automatically handles:
- Email reply chain stripping (`Am ... schrieb`, `Von:`, `From:`, `-----Original`, etc.)
- Email addresses → `[EMAIL]`
- IBAN numbers → `[IBAN]`
- German phone numbers (`+49` / `0` prefix) → `[PHONE]`
- German postal code + city patterns → `[LOCATION]`
- Salutation names (`Hallo Nicole,` → `Hallo [NAME],`)
- Skips subscription-cancellation tickets (sevdesk-meta, not Billy-relevant)

---

## Step 2: LLM Review Pass

After the script runs, read `sevdesk_tickets.json` and check for residual PII the regex missed.
Common cases in German CS text:

**Names in body text (not salutations)**
- "Frau Müller hat angerufen" → "Frau [NAME] hat angerufen"
- "Herr Schmidt aus Berlin" → "Herr [NAME] aus [LOCATION]"
- Full names in email signatures → remove the signature block entirely

**Personal company names**
- "Müller GmbH", "Becker & Partner Steuerberatung" → `[COMPANY]`
- Generic forms ("GmbH", "UG", "AG" alone) are fine — leave them

**Street addresses**
- "Musterstraße 12, 12345 Berlin" → `[ADDRESS]`

**Account/customer IDs that re-identify**
- Long numeric IDs appearing alongside other context → `[REF-XXXX]` (keep first 4 digits for category context)

**Agent signatures in ENGAGEMENT_MESSAGE**
- "Viele Grüße, Marzena / Customer Success Manager" → strip from `expected_answer`; keep the substantive reply text only

---

## Step 3: Domain Adaptation (deferred — first batch ships as `sevdesk_raw`)

First 50 tickets are committed with `"source": "sevdesk_raw"` — sevdesk product refs intact.
This lets us validate the harness before the adaptation pass.

When adapting a batch:
- `sevdesk` → `Billy`
- `DATEV-Export` → `CSV-Export` or remove
- `Hilfe.sevdesk.de` URLs → remove or replace with `[HELP_URL]`
- "in sevdesk unter Einstellungen" → "in Billy under Settings"
- Update `"source"` → `"sevdesk_adapted"`

---

## Placeholder Convention

| PII Type | Placeholder |
|---|---|
| Email address | `[EMAIL]` |
| Phone number | `[PHONE]` |
| Personal name | `[NAME]` |
| Personal company name | `[COMPANY]` |
| Street address | `[ADDRESS]` |
| German postal area | `[LOCATION]` |
| IBAN | `[IBAN]` |
| Ticket / account reference | `[REF-XXXX]` |

---

## Pre-Commit Checklist

Run this before `git add` on any fixture file derived from real data:

- [ ] No `@` character in any `query` or `expected_answer` value
- [ ] No raw phone numbers (7+ consecutive digits not part of a known ID format)
- [ ] No German postal codes (`\d{5}`) adjacent to city names
- [ ] No `Hallo [FirstName],` / `Liebe[r] [FirstName]` with real names
- [ ] No `DE\d{20}` IBAN patterns
- [ ] No personal-surname company names
- [ ] No agent full names in `expected_answer` (signature blocks stripped)
- [ ] `"source"` field present on every record (`sevdesk_raw` or `sevdesk_adapted`)
- [ ] Fixture file is in `.gitignore` or confirmed scrubbed before pushing

Quick grep to verify:
```bash
python3 -c "
import json, re, sys
data = json.load(open('tests/evalsuite/fixtures/sevdesk_tickets.json'))
issues = []
for r in data:
    for field in ('query', 'expected_answer'):
        v = r.get(field, '') or ''
        if '@' in v:           issues.append(f'{r[\"id\"]}.{field}: contains @')
        if re.search(r'\d{7,}', v): issues.append(f'{r[\"id\"]}.{field}: possible phone/ID')
        if re.search(r'DE\d{20}', v): issues.append(f'{r[\"id\"]}.{field}: possible IBAN')
if issues:
    print('PII CHECK FAILED:')
    for i in issues: print(' ', i)
    sys.exit(1)
else:
    print('PII check passed.')
"
```

---

## What NOT to scrub

- Generic business terms: "GmbH", "UG", "AG" used alone
- Product names: "sevdesk", "DATEV", "Billy", "Stripe"
- Category and intent labels
- Generic greetings: "Hallo", "Sehr geehrte Damen und Herren"
- CES ratings, timestamps, category strings
- Agent role titles without names: "Customer Success Manager", "Support Team"
