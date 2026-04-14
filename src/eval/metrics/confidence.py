"""Confidence gate calibration metrics.

Evaluates how well the CRAG confidence gate separates good retrievals
from poor ones, and whether the threshold is well-calibrated.

Threshold defaults from the librarian pipeline:
- CRAG gate: ``0.3``  (``GeneratorAgent.DEFAULT_CONFIDENCE_GATE``)
- Routing clarify: ``0.5`` (``QueryRouter.clarify_confidence_threshold``)
- Failure boundary: ``0.3`` / ``0.5`` (``FailureClusterer``)
"""

from __future__ import annotations

from dataclasses import dataclass, field


CRAG_GATE_THRESHOLD = 0.3
ROUTING_CLARIFY_THRESHOLD = 0.5


@dataclass
class GateMetrics:
    """Aggregate confidence gate calibration metrics."""

    accuracy: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    threshold: float = 0.0
    n_samples: int = 0


@dataclass
class CalibrationBin:
    """Single bin in a calibration curve."""

    bin_start: float
    bin_end: float
    avg_confidence: float
    avg_quality: float
    count: int


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def gate_accuracy(
    scores: list[float],
    truths: list[bool],
    threshold: float = CRAG_GATE_THRESHOLD,
) -> float:
    """Fraction of correct gate decisions.

    A gate decision is correct if:
    - ``score >= threshold`` and ``truth is True``  (correctly confident)
    - ``score < threshold`` and ``truth is False``   (correctly retried)

    Args:
        scores: Confidence scores from the pipeline.
        truths: Ground truth quality labels (True = good retrieval).
        threshold: Gate threshold to evaluate.
    """
    if not scores:
        return 0.0
    correct = sum(
        1
        for s, t in zip(scores, truths)
        if (s >= threshold) == t
    )
    return correct / len(scores)


def false_positive_rate(
    scores: list[float],
    truths: list[bool],
    threshold: float = CRAG_GATE_THRESHOLD,
) -> float:
    """Rate of unnecessary CRAG retries (gate says low-confidence but truth is good).

    FPR = false_positives / (false_positives + true_negatives)
    where "positive" means the gate triggers a retry (score < threshold).
    """
    negatives = [(s, t) for s, t in zip(scores, truths) if t is True]
    if not negatives:
        return 0.0
    false_pos = sum(1 for s, _ in negatives if s < threshold)
    return false_pos / len(negatives)


def false_negative_rate(
    scores: list[float],
    truths: list[bool],
    threshold: float = CRAG_GATE_THRESHOLD,
) -> float:
    """Rate of missed low-quality results (gate says confident but truth is bad).

    FNR = false_negatives / (false_negatives + true_positives)
    where "negative" means the gate passes (score >= threshold).
    """
    positives = [(s, t) for s, t in zip(scores, truths) if t is False]
    if not positives:
        return 0.0
    false_neg = sum(1 for s, _ in positives if s >= threshold)
    return false_neg / len(positives)


def optimal_threshold(
    scores: list[float],
    truths: list[bool],
    steps: int = 100,
) -> tuple[float, float]:
    """Find the threshold that maximizes F1 score.

    Sweeps thresholds in ``[0, 1]`` and returns ``(best_threshold, best_f1)``.
    """
    if not scores:
        return 0.0, 0.0

    best_t = 0.0
    best_f1 = 0.0

    for i in range(steps + 1):
        t = i / steps
        tp = sum(1 for s, tr in zip(scores, truths) if s >= t and tr is True)
        fp = sum(1 for s, tr in zip(scores, truths) if s >= t and tr is False)
        fn = sum(1 for s, tr in zip(scores, truths) if s < t and tr is True)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


def calibration_curve(
    scores: list[float],
    truths: list[bool],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Reliability diagram data for confidence calibration.

    Bins scores into ``n_bins`` equal-width buckets and computes
    average confidence vs. average quality (fraction of true labels)
    per bin.
    """
    if not scores:
        return []

    bins: list[CalibrationBin] = []
    bin_width = 1.0 / n_bins

    for i in range(n_bins):
        lo = i * bin_width
        hi = (i + 1) * bin_width

        bucket = [
            (s, t)
            for s, t in zip(scores, truths)
            if lo <= s < hi or (i == n_bins - 1 and s == hi)
        ]

        if not bucket:
            continue

        avg_conf = sum(s for s, _ in bucket) / len(bucket)
        avg_qual = sum(1 for _, t in bucket if t) / len(bucket)
        bins.append(
            CalibrationBin(
                bin_start=lo,
                bin_end=hi,
                avg_confidence=avg_conf,
                avg_quality=avg_qual,
                count=len(bucket),
            )
        )

    return bins


def evaluate_gate(
    scores: list[float],
    truths: list[bool],
    threshold: float = CRAG_GATE_THRESHOLD,
) -> GateMetrics:
    """Evaluate confidence gate calibration.

    Args:
        scores: Confidence scores from traced pipeline runs.
        truths: Ground truth quality labels (True = retrieval was good).
        threshold: Gate threshold to evaluate.

    Returns:
        ``GateMetrics`` with accuracy, FPR, FNR, and sample count.
    """
    return GateMetrics(
        accuracy=gate_accuracy(scores, truths, threshold),
        false_positive_rate=false_positive_rate(scores, truths, threshold),
        false_negative_rate=false_negative_rate(scores, truths, threshold),
        threshold=threshold,
        n_samples=len(scores),
    )
