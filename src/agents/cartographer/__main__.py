"""CLI entry point for the cartographer agent.

Usage:
    uv run cartographer --dry-run          # Extract stats only
    uv run cartographer                    # Full HTML report
    uv run cartographer --cron             # Cron-triggered analysis (SESSION.md + friction log)
"""

from __future__ import annotations

import sys


def main() -> None:
    """Route to parser (default) or cron subcommand."""
    if "--cron" in sys.argv:
        sys.argv.remove("--cron")
        from agents.cartographer.cron import run_cron

        run_cron()
    else:
        from agents.cartographer.parser import main as parser_main

        parser_main()


if __name__ == "__main__":
    main()
