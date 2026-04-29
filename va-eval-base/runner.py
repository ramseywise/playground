"""Evaluation runner: load fixtures, run harness, grade, report."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import structlog

from .graders import BaselineGrader, MessageQualityGrader, RoutingGrader, SchemaGrader
from .harness import run_eval_suite
from .metrics import OrchestrationMetricsGrader, RAGMetricsGrader
from .models import EvalReport, EvalTask, GraderResult, ServiceResponse

log = structlog.get_logger(__name__)


def load_clara_fixtures(fixture_path: Path | str | None = None) -> list[EvalTask]:
    """Load Clara tickets from va-langgraph test fixtures."""
    if fixture_path is None:
        fixture_path = (
            Path(__file__).parent.parent
            / "va-langgraph"
            / "tests"
            / "evalsuite"
            / "fixtures"
            / "clara_tickets.json"
        )

    fixture_path = Path(fixture_path)
    if not fixture_path.exists():
        raise FileNotFoundError(f"Clara fixtures not found at {fixture_path}")

    with open(fixture_path) as f:
        tickets = json.load(f)

    tasks = [EvalTask(**ticket) for ticket in tickets]
    log.info("loaded_clara_fixtures", n_tasks=len(tasks))
    return tasks


def _aggregate_by_service(
    results: list[GraderResult],
) -> dict[str, dict[str, Any]]:
    """Compute pass rates and scores by service."""
    by_service: dict[str, dict[str, Any]] = {}

    for result in results:
        if result.service not in by_service:
            by_service[result.service] = {
                "n_tasks": 0,
                "n_passed": 0,
                "total_score": 0.0,
                "by_grader": {},
            }

        svc = by_service[result.service]
        svc["n_tasks"] += 1
        svc["n_passed"] += int(result.is_correct)
        svc["total_score"] += result.score

        if result.grader_type not in svc["by_grader"]:
            svc["by_grader"][result.grader_type] = {
                "n_passed": 0,
                "n_tasks": 0,
                "total_score": 0.0,
            }

        grader_stats = svc["by_grader"][result.grader_type]
        grader_stats["n_tasks"] += 1
        grader_stats["n_passed"] += int(result.is_correct)
        grader_stats["total_score"] += result.score

    # Compute pass rates and averages
    for svc_data in by_service.values():
        svc_data["pass_rate"] = (
            svc_data["n_passed"] / svc_data["n_tasks"]
            if svc_data["n_tasks"] > 0
            else 0.0
        )
        svc_data["avg_score"] = (
            svc_data["total_score"] / svc_data["n_tasks"]
            if svc_data["n_tasks"] > 0
            else 0.0
        )

        for grader_stats in svc_data["by_grader"].values():
            grader_stats["pass_rate"] = (
                grader_stats["n_passed"] / grader_stats["n_tasks"]
                if grader_stats["n_tasks"] > 0
                else 0.0
            )
            grader_stats["avg_score"] = (
                grader_stats["total_score"] / grader_stats["n_tasks"]
                if grader_stats["n_tasks"] > 0
                else 0.0
            )

    return by_service


def _aggregate_by_grader(
    results: list[GraderResult],
) -> dict[str, dict[str, Any]]:
    """Compute pass rates and scores by grader type."""
    by_grader: dict[str, dict[str, Any]] = {}

    for result in results:
        if result.grader_type not in by_grader:
            by_grader[result.grader_type] = {
                "n_tasks": 0,
                "n_passed": 0,
                "total_score": 0.0,
                "by_service": {},
            }

        grader = by_grader[result.grader_type]
        grader["n_tasks"] += 1
        grader["n_passed"] += int(result.is_correct)
        grader["total_score"] += result.score

        if result.service not in grader["by_service"]:
            grader["by_service"][result.service] = {
                "n_passed": 0,
                "n_tasks": 0,
                "total_score": 0.0,
            }

        svc_stats = grader["by_service"][result.service]
        svc_stats["n_tasks"] += 1
        svc_stats["n_passed"] += int(result.is_correct)
        svc_stats["total_score"] += result.score

    # Compute pass rates and averages
    for grader_data in by_grader.values():
        grader_data["pass_rate"] = (
            grader_data["n_passed"] / grader_data["n_tasks"]
            if grader_data["n_tasks"] > 0
            else 0.0
        )
        grader_data["avg_score"] = (
            grader_data["total_score"] / grader_data["n_tasks"]
            if grader_data["n_tasks"] > 0
            else 0.0
        )

        for svc_stats in grader_data["by_service"].values():
            svc_stats["pass_rate"] = (
                svc_stats["n_passed"] / svc_stats["n_tasks"]
                if svc_stats["n_tasks"] > 0
                else 0.0
            )
            svc_stats["avg_score"] = (
                svc_stats["total_score"] / svc_stats["n_tasks"]
                if svc_stats["n_tasks"] > 0
                else 0.0
            )

    return by_grader


async def run_eval(
    run_name: str,
    tasks: list[EvalTask] | None = None,
    fixture_path: Path | str | None = None,
    baseline_only: bool = False,
) -> EvalReport:
    """Run full eval: load fixtures, call all services, grade, report."""
    if tasks is None:
        tasks = load_clara_fixtures(fixture_path)

    log.info("starting_eval", run_name=run_name, n_tasks=len(tasks))

    # Run all tasks against all services
    harness_results = await run_eval_suite(tasks)

    # Flatten responses for grading
    all_responses: dict[str, ServiceResponse] = {}
    for task_results in harness_results:
        for service, response in task_results.items():
            key = f"{response.task_id}:{service}"
            all_responses[key] = response

    # Initialize graders
    graders: list[BaselineGrader] = [
        SchemaGrader(),
        MessageQualityGrader(),
        RoutingGrader(),
    ]

    if not baseline_only:
        graders.extend(
            [
                RAGMetricsGrader(),
                OrchestrationMetricsGrader(),
            ]
        )

    # Grade all responses
    grader_tasks = []
    for task_results in harness_results:
        for service, response in task_results.items():
            task = next((t for t in tasks if t.id == response.task_id), None)
            if task:
                for grader in graders:
                    grader_tasks.append(grader.grade(task, response))

    grader_results = await asyncio.gather(*grader_tasks)

    # Aggregate results
    by_service = _aggregate_by_service(grader_results)
    by_grader = _aggregate_by_grader(grader_results)

    n_passed = sum(1 for r in grader_results if r.is_correct)
    avg_score = (
        sum(r.score for r in grader_results) / len(grader_results)
        if grader_results
        else 0.0
    )

    report = EvalReport(
        run_name=run_name,
        n_tasks=len(tasks),
        results=grader_results,
        by_service=by_service,
        by_grader=by_grader,
    )

    log.info(
        "eval_complete",
        run_name=run_name,
        n_results=len(grader_results),
        pass_rate=n_passed / len(grader_results) if grader_results else 0.0,
        avg_score=avg_score,
    )

    return report


def print_report(report: EvalReport) -> str:
    """Format EvalReport as a human-readable summary."""
    lines = [
        f"\n{'=' * 80}",
        f"Eval Report: {report.run_name}",
        f"Timestamp: {report.timestamp.isoformat()}",
        f"Run ID: {report.run_id}",
        f"{'=' * 80}\n",
    ]

    # Overall
    total_results = len(report.results)
    total_passed = sum(1 for r in report.results if r.is_correct)
    total_score = (
        sum(r.score for r in report.results) / total_results if total_results else 0.0
    )

    lines.append(
        f"Overall: {total_passed}/{total_results} passed ({100 * total_passed / total_results if total_results else 0:.1f}%)"
    )
    lines.append(f"Average Score: {total_score:.3f}\n")

    # By Service
    lines.append("By Service:")
    lines.append("-" * 80)
    for service, stats in sorted(report.by_service.items()):
        lines.append(f"  {service}")
        lines.append(
            f"    Pass Rate: {stats['n_passed']}/{stats['n_tasks']} ({100 * stats['pass_rate']:.1f}%)"
        )
        lines.append(f"    Avg Score: {stats['avg_score']:.3f}")
        for grader_type, grader_stats in stats["by_grader"].items():
            lines.append(
                f"      {grader_type}: {grader_stats['n_passed']}/{grader_stats['n_tasks']} "
                f"({100 * grader_stats['pass_rate']:.1f}%) — {grader_stats['avg_score']:.3f}"
            )
        lines.append("")

    # By Grader
    lines.append("\nBy Grader Type:")
    lines.append("-" * 80)
    for grader_type, stats in sorted(report.by_grader.items()):
        lines.append(f"  {grader_type}")
        lines.append(
            f"    Pass Rate: {stats['n_passed']}/{stats['n_tasks']} ({100 * stats['pass_rate']:.1f}%)"
        )
        lines.append(f"    Avg Score: {stats['avg_score']:.3f}")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)
