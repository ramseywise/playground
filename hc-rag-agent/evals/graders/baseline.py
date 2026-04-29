"""Recommended entry point for offline evaluation: QA graders + RAG component baselines.

**Graders (task-level)** — exact/MCQ, LLM grounding & friction, human review:

- :class:`ExactMatchGrader`, :class:`MCQGrader`
- :class:`GroundingJudge`, :class:`FrictionJudge`
- :class:`HumanGrader`, :exc:`PendingReviewError`

**RAG component metrics** — retrieval quality, reranker lift, confidence-gate calibration
(used for baseline / regression against golden retrieval data; not the same as answer-only LLM scores):

- Retrieval: :func:`evaluate_retrieval`, :func:`precision_at_k`, :func:`recall_at_k`, :func:`ndcg_at_k`
- Shared primitives: :class:`RetrievalHit`, :func:`compute_retrieval_hits`, :func:`aggregate_hit_rate`, :func:`aggregate_mrr`
- Reranker: :func:`evaluate_reranker`, :class:`RerankerMetrics`
- Confidence gate: :func:`evaluate_gate`, :class:`GateMetrics`

**Types** — :class:`~evals.utils.models.GoldenSample`, :class:`~evals.utils.models.RetrievalMetrics`

**Library answer metrics** (faithfulness / relevancy stacks): import :class:`~evals.graders.RagasGrader`
or :class:`~evals.graders.DeepEvalGrader` from :mod:`evals.graders`.
"""

from __future__ import annotations

from evals.graders.hitl import HumanGrader, PendingReviewError
from evals.graders.lexical import ExactMatchGrader
from evals.graders.llm_judge import FrictionJudge, GroundingJudge
from evals.graders.mcq import MCQGrader
from evals.metrics.confidence import GateMetrics, evaluate_gate
from evals.metrics.reranker import RerankerMetrics, evaluate_reranker
from evals.metrics.retrieval import (
    evaluate_retrieval,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from evals.metrics._shared import (
    RetrievalHit,
    RetrieveFn,
    aggregate_hit_rate,
    aggregate_mrr,
    compute_retrieval_hits,
)
from evals.utils.models import GoldenSample, RetrievalMetrics

__all__ = [
    # Graders
    "ExactMatchGrader",
    "MCQGrader",
    "GroundingJudge",
    "FrictionJudge",
    "HumanGrader",
    "PendingReviewError",
    # Retrieval baseline
    "RetrievalHit",
    "RetrieveFn",
    "aggregate_hit_rate",
    "aggregate_mrr",
    "compute_retrieval_hits",
    "evaluate_retrieval",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    # Reranker
    "RerankerMetrics",
    "evaluate_reranker",
    # Confidence gate
    "GateMetrics",
    "evaluate_gate",
    # Types
    "GoldenSample",
    "RetrievalMetrics",
]
