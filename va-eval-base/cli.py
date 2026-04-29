"""CLI for running baseline eval across all VA services."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click
import structlog

from . import run_eval, print_report

log = structlog.get_logger(__name__)


@click.command()
@click.option(
    "--name",
    "-n",
    default="baseline-eval",
    help="Name for this eval run",
)
@click.option(
    "--fixture-path",
    "-f",
    type=click.Path(exists=True),
    default=None,
    help="Path to Clara fixtures JSON (defaults to va-langgraph/tests/evalsuite/fixtures/clara_tickets.json)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Save full results JSON to this path",
)
@click.option(
    "--baseline-only",
    is_flag=True,
    help="Run only baseline graders (schema, message quality, routing)",
)
def main(name: str, fixture_path: str | None, output: str | None, baseline_only: bool):
    """Run eval suite across all VA services."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )

    click.echo(f"Running eval: {name}")
    click.echo(f"Baseline only: {baseline_only}")

    report = asyncio.run(
        run_eval(
            run_name=name,
            fixture_path=fixture_path,
            baseline_only=baseline_only,
        )
    )

    # Print human-readable summary
    click.echo(print_report(report))

    # Save JSON if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
        click.echo(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
