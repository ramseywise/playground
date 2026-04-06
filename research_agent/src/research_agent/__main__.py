from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog
from dotenv import load_dotenv

load_dotenv()

log = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Obsidian notes from a PDF using Claude."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the source PDF file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rendered note to stdout without writing files",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        metavar="SLUG",
        help="Override the Obsidian topic folder slug (e.g. 'rag', 'agentic-ai')",
    )
    args = parser.parse_args()

    if not args.pdf_path.exists():
        log.error("cli.pdf_not_found", path=str(args.pdf_path))
        print(f"Error: PDF not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    if not args.pdf_path.suffix.lower() == ".pdf":
        log.error("cli.not_a_pdf", path=str(args.pdf_path))
        print(f"Error: file is not a PDF: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    from agents.research_agent.src.research_agent.agent import ResearchAgent
    from agents.research_agent.src.research_agent.writer import render_note, write_note

    log.info("cli.start", pdf=str(args.pdf_path), dry_run=args.dry_run, topic=args.topic)

    try:
        agent = ResearchAgent()
        note = agent.process_pdf(args.pdf_path, topic_override=args.topic)

        if args.dry_run:
            print(render_note(note))
            log.info("cli.dry_run_done")
        else:
            note_path = write_note(note, args.pdf_path, topic_override=args.topic)
            log.info("cli.done", output=str(note_path))
            print(f"Note written: {note_path}")

    except FileExistsError as exc:
        log.error("cli.note_exists", error=str(exc))
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("cli.interrupted")
        sys.exit(130)
    except Exception as exc:
        log.error("cli.error", error=str(exc))
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
