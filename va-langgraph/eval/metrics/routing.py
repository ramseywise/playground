"""Routing metrics: macro precision/recall/F1 and per-intent breakdown."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import EvalTask, GraderResult


@dataclass
class RoutingMetrics:
    overall_f1: float
    overall_precision: float
    overall_recall: float
    per_intent: dict[str, dict[str, float]] = field(default_factory=dict)


def compute_routing_metrics(
    tasks: list[EvalTask],
    results: list[GraderResult],
) -> RoutingMetrics:
    """Compute macro-averaged precision/recall/F1 and per-intent breakdown.

    Pulls ground truth from task.expected_intent and predictions from
    task.metadata['classified_intent'] (set by the test harness before grading).
    """
    routing_results = {r.task_id: r for r in results if r.grader_type == "routing"}
    task_map = {t.id: t for t in tasks}

    true_labels: list[str] = []
    pred_labels: list[str] = []

    for task_id, r in routing_results.items():
        task = task_map.get(task_id)
        if task and task.expected_intent:
            true_labels.append(task.expected_intent)
            pred_labels.append(task.metadata.get("classified_intent", "__unknown__"))

    if not true_labels:
        return RoutingMetrics(0.0, 0.0, 0.0, {})

    intents = sorted(set(true_labels) | set(pred_labels))
    per_intent: dict[str, dict[str, float]] = {}

    for intent in intents:
        tp = sum(
            1 for t, p in zip(true_labels, pred_labels) if t == intent and p == intent
        )
        fp = sum(
            1 for t, p in zip(true_labels, pred_labels) if t != intent and p == intent
        )
        fn = sum(
            1 for t, p in zip(true_labels, pred_labels) if t == intent and p != intent
        )
        support = sum(1 for t in true_labels if t == intent)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        per_intent[intent] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": float(support),
        }

    macro_precision = sum(v["precision"] for v in per_intent.values()) / len(per_intent)
    macro_recall = sum(v["recall"] for v in per_intent.values()) / len(per_intent)
    macro_f1 = sum(v["f1"] for v in per_intent.values()) / len(per_intent)

    return RoutingMetrics(
        overall_f1=round(macro_f1, 3),
        overall_precision=round(macro_precision, 3),
        overall_recall=round(macro_recall, 3),
        per_intent=per_intent,
    )
