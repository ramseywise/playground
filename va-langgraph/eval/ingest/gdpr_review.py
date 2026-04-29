"""
LLM review pass for Clara eval fixtures — catches residual PII the regex missed.

COMPLIANCE: This script sends post-regex fixture data to an external LLM API (Gemini).
Post-regex output may still contain residual PII (names mid-sentence, personal company
names, personal URLs) — that is the purpose of this pass. Sending it externally requires:
  1. A signed Data Processing Agreement (DPA) with Google covering EU customer data, AND
  2. A documented legitimate interest assessment under GDPR Art. 6(1)(f).
Do not run in production environments without both in place.
See .claude/skills/gdpr-scrub/SKILL.md for a compliant alternative using a local model.

Run this after clara_ingest.py, review the findings, apply fixes manually,
then run the pre-commit grep check from .claude/skills/gdpr-scrub/SKILL.md.

Usage:
    cd va-langgraph
    uv run python eval/ingest/gdpr_review.py                        # full batch
    uv run python eval/ingest/gdpr_review.py --sample 14           # calibration run (first N fixtures)
    uv run python eval/ingest/gdpr_review.py --batch-size 30       # tune batch size
    uv run python eval/ingest/gdpr_review.py --findings findings.json  # save output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(Path(__file__).parents[3] / ".env")

_REPO_ROOT = Path(__file__).parents[3]
DEFAULT_INPUT = (
    _REPO_ROOT
    / "va-langgraph"
    / "tests"
    / "evalsuite"
    / "fixtures"
    / "clara_tickets.json"
)

# ---------------------------------------------------------------------------
# Review prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a GDPR compliance reviewer for a German-language customer support dataset.
Your task is to identify residual personally identifiable information (PII) in
eval fixture records that have already been through a regex scrubbing pass.

The regex pass has already replaced:
  [EMAIL]    — email addresses
  [PHONE]    — phone numbers (German +49 and Austrian +43 formats)
  [IBAN]     — German IBANs (DE + 20 digits)
  [LOCATION] — German/Austrian postal code + city (e.g. "12345 Berlin", "4020 Linz")
  [NAME]     — names following salutations (Hallo, Guten Tag, Liebe/r, Dear, Sehr geehrte/r)
               AND names following Herr/Frau in body text
  [ADDRESS]  — street addresses (Musterstraße 12, Hauptstr. 5a, Am Nelkenberg 13)
  [REF]      — ticket/order reference numbers (7+ digit IDs, Rechnung Nr. XXXXX, B2B-XXXXXX)
  [BIZ-ID]   — business registration numbers (UID, ATU, USt-IdNr, FN, HRA, HRB)
  [COMPANY]  — personal company names flagged in prior review

You must flag ONLY what the regex still misses. Based on prior review of this dataset,
the most common residual gaps are:

1. PERSONAL NAMES not following Herr/Frau or a salutation:
   - Full names referenced mid-sentence: "Klaus Meier hat angerufen" → flag "Klaus Meier", suggest "[NAME]"
   - Agent first names in signature remnants: "Dein Felix" → flag "Felix", suggest "[NAME]"
   - Surnames alone after "[NAME]" replacement: "[NAME] Müller" → flag "Müller", suggest "[NAME]"

2. PERSONAL COMPANY NAMES that identify individuals:
   - Distinctive company names: "Müller GmbH", "Becker & Partner Steuerberatung" → suggest "[COMPANY]"
   - "Fa. Erich Wutzig" → flag full name including Fa., suggest "[COMPANY]"
   - Generic legal forms alone ("GmbH", "UG", "AG") are FINE — do not flag

3. STREET ADDRESSES not caught by regex (unusual formats):
   - Short street abbreviations: "Bahnhofstr. 14" where str. is not at end of word
   - Austrian address formats: "Wienerstraße 59 / Top 2" — the / Top part

4. ACCOUNT / CUSTOMER IDs alongside personal context:
   - Short numeric IDs (6 digits) with keywords like "Kundennummer 368488" → suggest "[REF]"
   - Commercial register numbers not caught: "HRA 6308" → suggest "[BIZ-ID]"

5. PERSONAL URLS (individual's own website, not product/service sites):
   - "www.erikbender.de" (person's website) → suggest "[URL]"
   - Do NOT flag: sevdesk.de, billing sites, help.sevdesk.de, known SaaS domains

6. ANY REMAINING EMAIL PATTERNS:
   - If @ appears and was not caught → flag

DO NOT flag:
  - Already-replaced placeholders: [EMAIL], [NAME], [REF], [ADDRESS], [COMPANY], [BIZ-ID], etc.
    - IMPORTANT: "Hallo [NAME] [NAME]," is fine — first + last name both correctly replaced
    - IMPORTANT: "liebe [NAME]\n[NAME]" is fine — greeting + closing signature both replaced
    - IMPORTANT: two adjacent [NAME] tokens anywhere means both were already scrubbed
  - Product names: sevdesk, Billy, DATEV, Stripe, FinAPI, HubSpot, Revolut
  - Generic business terms: GmbH, UG, AG, GbR, e.V., StB, WP used alone (no personal name)
  - Role titles without names: "Customer Success Manager", "Support Team", "Ihr Team", "StB"
  - Generic greetings: "Hallo", "Sehr geehrte Damen und Herren"
  - CES ratings, category labels, timestamps
  - Numbers inside URLs or article IDs (e.g. /articles/9453113-)
  - German VAT rates (19%, 7%), generic invoice numbers without personal context
  - Known error ID formats: UUID patterns like "a1c16c1e-92e9-43f0-98b6-..."
  - Tracking tokens inside URLs (the URL itself is the PII vector, not the token)
  - Public GitHub repository URLs — these are shared technical references, not personal PII
  - Public SaaS domains: news.sevdesk.de, help.sevdesk.de, app.sevdesk.de, stripe.com, etc.

Output format — respond with a JSON array only, no prose:
[
  {
    "id": "sev-042",
    "field": "query" | "expected_answer",
    "issue_type": "name" | "company" | "address" | "account_id" | "ip" | "email" | "other",
    "snippet": "exact text fragment containing the PII (≤60 chars)",
    "suggestion": "replacement text or action",
    "confidence": "high" | "medium" | "low"
  }
]

If no issues are found in a batch, return an empty array: []
"""

USER_PROMPT_TEMPLATE = """\
Review the following {n} fixture records for residual PII.
Each record has an "id", "query", and "expected_answer" field.
Check both fields of every record. Return only the JSON findings array.

RECORDS:
{records_json}
"""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _review_batch(llm: ChatGoogleGenerativeAI, batch: list[dict]) -> list[dict]:
    slim = [
        {"id": f["id"], "query": f["query"], "expected_answer": f["expected_answer"]}
        for f in batch
    ]
    user_msg = USER_PROMPT_TEMPLATE.format(
        n=len(slim), records_json=json.dumps(slim, ensure_ascii=False, indent=2)
    )

    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)]
    )
    raw = response.content.strip()

    # Strip markdown code fences if the model wraps in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        findings = json.loads(raw)
    except json.JSONDecodeError:
        print(
            "  WARNING: could not parse LLM response as JSON — raw output saved",
            file=sys.stderr,
        )
        findings = [{"parse_error": True, "raw": raw}]

    return findings


def main(
    input_path: Path,
    findings_path: Path | None,
    sample: int | None,
    batch_size: int,
    model: str,
) -> None:
    if not input_path.exists():
        print(
            f"ERROR: {input_path} not found — run clara_ingest.py first",
            file=sys.stderr,
        )
        sys.exit(1)

    fixtures = json.loads(input_path.read_text())
    if sample:
        fixtures = fixtures[:sample]
        print(
            f"Calibration mode: reviewing first {len(fixtures)} fixtures",
            file=sys.stderr,
        )
    else:
        print(f"Full review: {len(fixtures)} fixtures", file=sys.stderr)

    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    all_findings: list[dict] = []

    batches = [
        fixtures[i : i + batch_size] for i in range(0, len(fixtures), batch_size)
    ]
    for idx, batch in enumerate(batches, 1):
        print(
            f"  Batch {idx}/{len(batches)} ({len(batch)} fixtures)...",
            file=sys.stderr,
            end=" ",
        )
        findings = _review_batch(llm, batch)
        all_findings.extend(findings)
        print(f"{len(findings)} findings", file=sys.stderr)

    # Summary
    print(f"\nTotal findings: {len(all_findings)}", file=sys.stderr)
    by_type = {}
    for f in all_findings:
        t = f.get("issue_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    if by_type:
        print(f"By type: {dict(sorted(by_type.items()))}", file=sys.stderr)

    high = [f for f in all_findings if f.get("confidence") == "high"]
    if high:
        print(f"\nHigh-confidence findings ({len(high)}):", file=sys.stderr)
        for f in high:
            print(
                f"  {f.get('id')} [{f.get('field')}] {f.get('issue_type')}: {f.get('snippet', '')!r}",
                file=sys.stderr,
            )

    output = json.dumps(all_findings, ensure_ascii=False, indent=2)
    if findings_path:
        findings_path.write_text(output)
        print(f"\nFindings written → {findings_path}", file=sys.stderr)
    else:
        print("\n--- FINDINGS ---")
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--findings", type=Path, default=None, help="Write findings JSON to this path"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Review only first N fixtures (calibration)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Fixtures per LLM call (default: 20)"
    )
    parser.add_argument(
        "--model", default="gemini-2.5-flash", help="Model to use for review"
    )
    args = parser.parse_args()
    main(args.input, args.findings, args.sample, args.batch_size, args.model)
