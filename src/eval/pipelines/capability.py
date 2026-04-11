"""Capability evaluation pipeline.

Runs tasks through graders, clusters failures, and builds an attribution
taxonomy.  Produces an EvalReport with pass metrics and breakdowns.

Flow: tasks -> graders -> failure clustering -> attribution -> EvalReport
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import structlog

from eval.models import (
    CategoryBreakdown,
    EvalReport,
    EvalRunConfig,
    EvalTask,
    GraderResult,
)
from librarian.tasks.tracing import FailureClusterer, PipelineTrace, PipelineTracer

log = structlog.get_logger(__name__)


async def run_capability_eval(
    tasks: list[EvalTask],
    graders: list[Any],
    *,
    config: EvalRunConfig | None = None,
    clusterer: FailureClusterer | None = None,
) -> EvalReport:
    """Run capability evaluation: each task through each grader.

    Args:
        tasks: List of evaluation tasks.
        graders: List of objects implementing the Grader protocol.
        config: Optional run configuration for reproducibility.
        clusterer: Optional failure clusterer (uses default if None).

    Returns:
        EvalReport with pass rates, breakdowns, and failure clusters.
    """
    if not tasks:
        msg = "No tasks provided for capability evaluation"
        raise ValueError(msg)
    if not graders:
        msg = "No graders provided for capability evaluation"
        raise ValueError(msg)

    config = config or EvalRunConfig()
    clusterer = clusterer or FailureClusterer()
    tracer = PipelineTracer()

    all_results: list[GraderResult] = []

    for task in tasks:
        trace = tracer.create_trace(task.id, task.query)
        task_results = await _grade_task(task, graders, trace)
        all_results.extend(task_results)

    clusters = clusterer.cluster_failures(tracer.get_failure_traces())

    return _build_report(
        results=all_results,
        tasks=tasks,
        graders=graders,
        config=config,
        clusters=clusters,
    )


async def _grade_task(
    task: EvalTask,
    graders: list[Any],
    trace: PipelineTrace,
) -> list[GraderResult]:
    """Run all graders on a single task, updating the trace."""
    results: list[GraderResult] = []
    any_passed = False

    for grader in graders:
        try:
            result = await grader.grade(task)
            results.append(result)
            if result.is_correct:
                any_passed = True
        except Exception as exc:
            log.warning(
                "capability.grade_error",
                task_id=task.id,
                grader=grader.grader_type,
                error=str(exc),
            )
            results.append(
                GraderResult(
                    task_id=task.id,
                    grader_type=grader.grader_type,
                    is_correct=False,
                    score=0.0,
                    reasoning=f"Grader error: {exc}",
                )
            )

    trace.status = "success" if any_passed else "failure"
    trace.confidence = max((r.score for r in results), default=0.0)
    if not any_passed:
        trace.failure_reason = "all_graders_failed"

    return results


def _build_report(
    results: list[GraderResult],
    tasks: list[EvalTask],
    graders: list[Any],
    config: EvalRunConfig,
    clusters: list[Any],
) -> EvalReport:
    """Aggregate results into an EvalReport."""
    n_tasks = len(tasks)
    passed_tasks = _count_passed_tasks(results, tasks)
    pass_rate = passed_tasks / n_tasks if n_tasks else 0.0
    avg_score = sum(r.score for r in results) / len(results) if results else 0.0

    by_category = _breakdown_by(results, tasks, key="category")
    by_difficulty = _breakdown_by(results, tasks, key="difficulty")
    by_grader = _breakdown_by_grader(results, graders)

    return EvalReport(
        run_id=config.run_name or "capability_eval",
        config=config,
        results=results,
        pass_rate=pass_rate,
        avg_score=avg_score,
        n_tasks=n_tasks,
        n_passed=passed_tasks,
        by_category=by_category,
        by_difficulty=by_difficulty,
        by_grader=by_grader,
        failure_clusters=[
            {"type": c.failure_type, "count": c.count, "patterns": c.common_patterns}
            for c in clusters
        ],
    )


def _count_passed_tasks(results: list[GraderResult], tasks: list[EvalTask]) -> int:
    """A task passes if ANY grader marks it correct."""
    passed_ids: set[str] = set()
    for r in results:
        if r.is_correct:
            passed_ids.add(r.task_id)
    return len(passed_ids & {t.id for t in tasks})


def _breakdown_by(
    results: list[GraderResult],
    tasks: list[EvalTask],
    key: str,
) -> list[CategoryBreakdown]:
    """Group results by a task attribute (category or difficulty)."""
    task_map = {t.id: t for t in tasks}
    groups: dict[str, list[GraderResult]] = defaultdict(list)
    for r in results:
        task = task_map.get(r.task_id)
        label = getattr(task, key, "unknown") if task else "unknown"
        groups[label].append(r)

    breakdowns = []
    for label, group_results in sorted(groups.items()):
        task_ids = {r.task_id for r in group_results}
        passed = sum(1 for tid in task_ids if any(r.is_correct for r in group_results if r.task_id == tid))
        breakdowns.append(
            CategoryBreakdown(
                category=label,
                n_tasks=len(task_ids),
                pass_rate=passed / len(task_ids) if task_ids else 0.0,
                avg_score=sum(r.score for r in group_results) / len(group_results),
            )
        )
    return breakdowns


def _breakdown_by_grader(
    results: list[GraderResult],
    graders: list[Any],
) -> list[CategoryBreakdown]:
    """Group results by grader type."""
    groups: dict[str, list[GraderResult]] = defaultdict(list)
    for r in results:
        groups[r.grader_type].append(r)

    breakdowns = []
    for grader_type, group_results in sorted(groups.items()):
        n_correct = sum(1 for r in group_results if r.is_correct)
        breakdowns.append(
            CategoryBreakdown(
                category=grader_type,
                n_tasks=len(group_results),
                pass_rate=n_correct / len(group_results) if group_results else 0.0,
                avg_score=sum(r.score for r in group_results) / len(group_results),
            )
        )
    return breakdowns
