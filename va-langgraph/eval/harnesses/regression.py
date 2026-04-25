"""Regression harness: floor-check assertions over deterministic graders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import EvalReport, EvalRunConfig, EvalTask
from ..runner import EvalRunner


@dataclass
class RegressionThresholds:
    routing_f1_floor: float = 0.85
    per_intent_precision_floor: float = 0.75
    injection_fnr_ceiling: float = 0.05
    pii_coverage_floor: float = 0.95
    schema_compliance_floor: float = 0.95


async def run_regression_eval(
    tasks: list[EvalTask],
    graders: list[Any],
    thresholds: RegressionThresholds | None = None,
    config: EvalRunConfig | None = None,
) -> EvalReport:
    """Run regression evaluation and return report.

    Callers are responsible for asserting metric floors against the returned
    report — typically via compute_routing_metrics() or compute_safety_metrics().
    """
    thresholds = thresholds or RegressionThresholds()
    runner = EvalRunner(graders=graders, config=config)
    return await runner.run_capability(tasks)
