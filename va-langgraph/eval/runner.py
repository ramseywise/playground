"""EvalRunner — orchestrates tasks × graders → EvalReport."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import CategoryBreakdown, EvalReport, EvalRunConfig, EvalTask, GraderResult


class EvalRunner:
    def __init__(
        self,
        graders: list[Any],
        config: EvalRunConfig | None = None,
    ) -> None:
        self.graders = graders
        self.config = config or EvalRunConfig(run_name="unnamed")

    async def run_capability(self, tasks: list[EvalTask]) -> EvalReport:
        """Run all tasks through all graders. A task passes if ANY grader passes."""
        all_results: list[GraderResult] = []

        for task in tasks:
            for grader in self.graders:
                result = await grader.grade(task)
                all_results.append(result)

        return self._build_report(tasks, all_results)

    def _build_report(
        self,
        tasks: list[EvalTask],
        results: list[GraderResult],
    ) -> EvalReport:
        task_passed: dict[str, bool] = {
            task.id: any(r.is_correct for r in results if r.task_id == task.id)
            for task in tasks
        }

        n_tasks = len(tasks)
        n_passed = sum(task_passed.values())
        pass_rate = n_passed / n_tasks if n_tasks else 0.0
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0

        failures = [
            {
                "task_id": t.id,
                "query": t.query,
                "category": t.category,
                "results": [
                    r.model_dump()
                    for r in results
                    if r.task_id == t.id and not r.is_correct
                ],
            }
            for t in tasks
            if not task_passed.get(t.id)
        ]

        return EvalReport(
            config=self.config,
            results=results,
            pass_rate=pass_rate,
            avg_score=avg_score,
            n_tasks=n_tasks,
            n_passed=n_passed,
            by_category=self._breakdown_by(
                tasks, results, task_passed, lambda t: t.category
            ),
            by_grader=self._breakdown_by_grader(results),
            failure_details=failures,
        )

    def _breakdown_by(
        self,
        tasks: list[EvalTask],
        results: list[GraderResult],
        task_passed: dict[str, bool],
        key_fn: Any,
    ) -> list[CategoryBreakdown]:
        groups: dict[str, list[EvalTask]] = defaultdict(list)
        for task in tasks:
            groups[key_fn(task)].append(task)

        breakdowns = []
        for cat, cat_tasks in sorted(groups.items()):
            cat_results = [r for t in cat_tasks for r in results if r.task_id == t.id]
            n = len(cat_tasks)
            n_pass = sum(task_passed.get(t.id, False) for t in cat_tasks)
            avg = (
                sum(r.score for r in cat_results) / len(cat_results)
                if cat_results
                else 0.0
            )
            breakdowns.append(
                CategoryBreakdown(
                    category=cat,
                    n_tasks=n,
                    n_passed=n_pass,
                    pass_rate=n_pass / n if n else 0.0,
                    avg_score=round(avg, 3),
                )
            )
        return breakdowns

    def _breakdown_by_grader(
        self, results: list[GraderResult]
    ) -> list[CategoryBreakdown]:
        groups: dict[str, list[GraderResult]] = defaultdict(list)
        for r in results:
            groups[r.grader_type].append(r)

        breakdowns = []
        for grader_type, grader_results in sorted(groups.items()):
            n = len(grader_results)
            n_pass = sum(r.is_correct for r in grader_results)
            avg = sum(r.score for r in grader_results) / n if n else 0.0
            breakdowns.append(
                CategoryBreakdown(
                    category=grader_type,
                    n_tasks=n,
                    n_passed=n_pass,
                    pass_rate=n_pass / n if n else 0.0,
                    avg_score=round(avg, 3),
                )
            )
        return breakdowns
