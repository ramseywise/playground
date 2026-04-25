"""
LLM review pass for sevdesk eval fixtures — catches residual PII the regex missed.

Run this after sevdesk_ingest.py, review the findings, apply fixes manually,
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
DEFAULT_INPUT = _REPO_ROOT / "va-langgraph" / "tests" / "evalsuite" / "fixtures" / "sevdesk_tickets.json"

# ---------------------------------------------------------------------------
# Review prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a GDPR compliance reviewer for a German-language customer support dataset.
Your task is to identify residual personally identifiable information (PII) in
eval fixture records that have already been through a regex scrubbing pass.

The regex pass has already replaced:
  [EMAIL]    — email addresses
  [PHONE]    — phone numbers
  [IBAN]     — German IBANs (DE + 20 digits)
  [LOCATION] — German postal code + city (e.g. "12345 Berlin")
  [NAME]     — first names following salutations (Hallo, Guten Tag, Liebe/r, Dear, Sehr geehrte/r)
  [REF]      — ticket reference numbers (7+ digit IDs)
  [BIZ-ID]   — business registration numbers (UID, ATU, USt-IdNr, FN)

You must flag ONLY what the regex missed. Common gaps in German CS text:

1. PERSONAL NAMES in body text (not salutations):
   - "Frau Müller hat angerufen" → flag, suggest "[NAME]"
   - "Herr Schmidt aus Berlin" → flag name + city
   - Agent full names in closing lines the signature stripper missed
     e.g. "Dein Max aus dem Support-Team" → flag "Max"

2. PERSONAL COMPANY NAMES that identify individuals:
   - "Müller GmbH", "Becker & Partner Steuerberatung" → flag, suggest "[COMPANY]"
   - Generic legal forms alone ("GmbH", "UG", "AG") are FINE — do not flag

3. STREET ADDRESSES:
   - "Musterstraße 12" or "Hauptstr. 5a" → flag, suggest "[ADDRESS]"
   - City names alone (without street) are fine

4. ACCOUNT / CUSTOMER IDs that re-identify when combined with context:
   - Long numeric IDs (8+ digits) that are NOT already [REF] and appear alongside
     personal context → flag, suggest "[REF]"

5. IP ADDRESSES that identify individuals (not known service IPs):
   - Personal router IPs → flag, suggest "[IP]"
   - Known sevdesk/service IPs (e.g. 167.89.85.240) → do NOT flag

6. ANY REMAINING EMAIL PATTERNS:
   - If @ appears and was not caught → flag

DO NOT flag:
  - Already-replaced placeholders like [EMAIL], [NAME], [REF], etc.
  - Product names: sevdesk, Billy, DATEV, Stripe, FinAPI, HubSpot
  - Generic business terms: GmbH, UG, AG, GbR used alone
  - Role titles without names: "Customer Success Manager", "Support Team", "Ihr Team"
  - Generic greetings: "Hallo", "Sehr geehrte Damen und Herren"
  - CES ratings, category labels, timestamps
  - Numbers inside URLs or article IDs (e.g. /articles/9453113-)
  - German VAT rates (19%, 7%) or invoice numbers without personal context

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
    slim = [{"id": f["id"], "query": f["query"], "expected_answer": f["expected_answer"]} for f in batch]
    user_msg = USER_PROMPT_TEMPLATE.format(n=len(slim), records_json=json.dumps(slim, ensure_ascii=False, indent=2))

    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)])
    raw = response.content.strip()

    # Strip markdown code fences if the model wraps in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        findings = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  WARNING: could not parse LLM response as JSON — raw output saved", file=sys.stderr)
        findings = [{"parse_error": True, "raw": raw}]

    return findings


def main(input_path: Path, findings_path: Path | None, sample: int | None, batch_size: int, model: str) -> None:
    if not input_path.exists():
        print(f"ERROR: {input_path} not found — run sevdesk_ingest.py first", file=sys.stderr)
        sys.exit(1)

    fixtures = json.loads(input_path.read_text())
    if sample:
        fixtures = fixtures[:sample]
        print(f"Calibration mode: reviewing first {len(fixtures)} fixtures", file=sys.stderr)
    else:
        print(f"Full review: {len(fixtures)} fixtures", file=sys.stderr)

    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    all_findings: list[dict] = []

    batches = [fixtures[i : i + batch_size] for i in range(0, len(fixtures), batch_size)]
    for idx, batch in enumerate(batches, 1):
        print(f"  Batch {idx}/{len(batches)} ({len(batch)} fixtures)...", file=sys.stderr, end=" ")
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
            print(f"  {f.get('id')} [{f.get('field')}] {f.get('issue_type')}: {f.get('snippet', '')!r}", file=sys.stderr)

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
    parser.add_argument("--findings", type=Path, default=None, help="Write findings JSON to this path")
    parser.add_argument("--sample", type=int, default=None, help="Review only first N fixtures (calibration)")
    parser.add_argument("--batch-size", type=int, default=20, help="Fixtures per LLM call (default: 20)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model to use for review")
    args = parser.parse_args()
    main(args.input, args.findings, args.sample, args.batch_size, args.model)
