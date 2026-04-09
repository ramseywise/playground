"""Cron-triggered insights analysis using the Anthropic SDK.

Reads session artifacts (SESSION.md, friction-log.jsonl, existing skills)
and uses Claude to identify workflow patterns and suggest new skills.

Run manually:
    uv run python scripts/insights_cron.py

Or schedule via Claude Code remote trigger / system cron.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import structlog

log = structlog.get_logger(__name__)

# --- Paths ---

REPO_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_DIR = REPO_ROOT / ".claude"
SESSION_FILE = CLAUDE_DIR / "docs" / "SESSION.md"
FRICTION_LOG = CLAUDE_DIR / "friction-log.jsonl"
SKILLS_DIR = CLAUDE_DIR / "skills"
INSIGHTS_DIR = CLAUDE_DIR / "docs" / "insights"
COMMANDS_DIR = CLAUDE_DIR / "commands"


def _read_if_exists(path: Path) -> str:
    """Read file contents or return empty string."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _read_friction_log(path: Path, max_lines: int = 200) -> str:
    """Read last N lines of friction log."""
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return "\n".join(lines[-max_lines:])


def _list_skills(skills_dir: Path) -> str:
    """List existing skill names and descriptions."""
    skills: list[str] = []
    for f in sorted(skills_dir.glob("*.md")):
        content = f.read_text(encoding="utf-8")
        name = f.stem
        # Extract description from frontmatter
        desc = ""
        if content.startswith("---"):
            end = content.find("---", 3)
            if end > 0:
                for line in content[3:end].splitlines():
                    if line.strip().startswith("description:"):
                        desc = line.split(":", 1)[1].strip().strip('"')
                        break
        skills.append(f"- {name}: {desc}")
    return "\n".join(skills)


def build_analysis_prompt(
    session_md: str,
    friction_log: str,
    existing_skills: str,
) -> str:
    """Build the analysis prompt for Claude."""
    return f"""Analyze these Claude Code session artifacts and produce two outputs:

## 1. Workflow Insights (top 3-5 patterns)

For each pattern:
- **Signal**: what the data shows
- **Interpretation**: what this likely means
- **Recommendation**: one concrete change

## 2. Skill Suggestions

Review the skill candidates from SESSION.md and friction log patterns.
Compare against existing skills to avoid duplicates.

For each candidate:
- **Verdict**: GENERATE | SKIP | MERGE
- **Reason**: one sentence
- If GENERATE: provide the full skill .md file content (with frontmatter)

---

## SESSION.md
```
{session_md}
```

## Friction log (recent entries)
```
{friction_log}
```

## Existing skills
```
{existing_skills}
```

---

Output as structured markdown. For any GENERATE verdict, include the complete
skill file content in a fenced code block with the suggested filename.
Keep analysis terse and actionable."""


def run_analysis() -> str:
    """Run the insights analysis via Anthropic SDK."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.error("insights.no_api_key")
        sys.exit(1)

    session_md = _read_if_exists(SESSION_FILE)
    friction_log = _read_friction_log(FRICTION_LOG)
    existing_skills = _list_skills(SKILLS_DIR)

    if not session_md and not friction_log:
        log.info("insights.no_data", msg="No session or friction data to analyze")
        return "No session data available for analysis."

    client = anthropic.Anthropic(api_key=api_key)

    prompt = build_analysis_prompt(session_md, friction_log, existing_skills)

    log.info("insights.calling_api", model="claude-sonnet-4-6")
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text
    log.info(
        "insights.done",
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )
    return response_text


def save_report(report: str) -> Path:
    """Save insights report to .claude/docs/insights/."""
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    out_path = INSIGHTS_DIR / f"{date_str}.md"

    # Append if same-day report exists
    if out_path.exists():
        existing = out_path.read_text(encoding="utf-8")
        report = f"{existing}\n\n---\n\n# Run {datetime.now(tz=timezone.utc).strftime('%H:%M UTC')}\n\n{report}"

    out_path.write_text(f"# Insights — {date_str}\n\n{report}\n", encoding="utf-8")
    log.info("insights.saved", path=str(out_path))
    return out_path


def extract_and_write_skills(report: str) -> list[str]:
    """Parse GENERATE skill blocks from report and write skill files.

    Returns list of created skill filenames.
    """
    created: list[str] = []
    lines = report.splitlines()
    i = 0
    while i < len(lines):
        # Look for filename hints like: `skills/foo_bar.md` or **Filename**: `foo_bar.md`
        line = lines[i]
        if "GENERATE" in line or ("filename" in line.lower() and ".md" in line):
            # Scan ahead for fenced code block
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            if j < len(lines):
                # Found opening fence
                k = j + 1
                content_lines: list[str] = []
                while k < len(lines) and not lines[k].strip().startswith("```"):
                    content_lines.append(lines[k])
                    k += 1
                content = "\n".join(content_lines).strip()
                if content.startswith("---") and "name:" in content:
                    # Extract filename from content frontmatter
                    for cl in content_lines:
                        if cl.strip().startswith("name:"):
                            name = cl.split(":", 1)[1].strip().strip('"').replace(" ", "_")
                            filename = f"{name}.md"
                            skill_path = SKILLS_DIR / filename
                            skill_path.write_text(
                                content + "\n", encoding="utf-8"
                            )
                            created.append(filename)
                            log.info("insights.skill_created", file=filename)
                            break
                i = k + 1
                continue
        i += 1
    return created


def main() -> None:
    """Entry point for cron-triggered insights."""
    log.info("insights.start")
    report = run_analysis()
    out_path = save_report(report)
    created_skills = extract_and_write_skills(report)

    summary = {
        "report": str(out_path),
        "skills_created": created_skills,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    # Write machine-readable summary for cron monitoring
    summary_path = INSIGHTS_DIR / "latest.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("insights.complete", **summary)


if __name__ == "__main__":
    main()
