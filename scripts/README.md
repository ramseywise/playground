# Scripts

## insights_cron.py

Cron-triggered workflow insights using the Anthropic SDK. Reads session artifacts and generates skill suggestions.

### Run manually

```bash
uv run python scripts/insights_cron.py
```

### Schedule options

**Option 1: Claude Code remote trigger** (preferred when authenticated)

```bash
# Via /schedule command in Claude Code:
/schedule create --name "weekly-insights" --cron "0 9 * * 1" --prompt "Run /insights and save the report"
```

**Option 2: System cron**

```cron
# Weekly Monday 9am — run insights analysis
0 9 * * 1 cd /path/to/repo && ANTHROPIC_API_KEY=... uv run python scripts/insights_cron.py
```

**Option 3: Session cron** (active only during Claude Code sessions)

Set up at the start of a session — expires when the session ends or after 7 days.
