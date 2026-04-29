"""Pre-commit PII grep check — run before git add on any fixture file from real data."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[3]
DEFAULT_FIXTURE = (
    _REPO_ROOT
    / "va-langgraph"
    / "tests"
    / "evalsuite"
    / "fixtures"
    / "clara_tickets.json"
)


def main(fixture_path: Path) -> None:
    if not fixture_path.exists():
        print(f"ERROR: {fixture_path} not found", file=sys.stderr)
        sys.exit(1)

    data = json.loads(fixture_path.read_text())
    issues: list[str] = []

    for r in data:
        for field in ("query", "expected_answer"):
            v = r.get(field, "") or ""
            # Strip URLs and UUID/error-id patterns before digit check to avoid false positives
            v_clean = re.sub(r"https?://\S+", "", v)
            v_clean = re.sub(
                r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
                "",
                v_clean,
                flags=re.IGNORECASE,
            )

            if "@" in v:
                issues.append(f"{r['id']}.{field}: contains @")
            if re.search(r"DE\d{20}", v):
                issues.append(f"{r['id']}.{field}: possible IBAN")
            if re.search(r"\d{7,}", v_clean):
                issues.append(f"{r['id']}.{field}: possible phone/ID (7+ digits)")

    if issues:
        print("PII check FAILED:")
        for i in issues:
            print(f"  {i}")
        sys.exit(1)
    else:
        print(f"PII check passed — {len(data)} fixtures clean.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args()
    main(args.input)
