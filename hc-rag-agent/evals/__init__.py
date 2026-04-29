"""Offline evaluation suite: graders, metrics, harnesses, experiments.

Structure:
- ``evals.graders``   — item-level verdict (LLM judge, exact match, MCQ, HITL)
- ``evals.metrics``   — dataset-level aggregates (retrieval hit_rate/MRR, reranker, confidence gate)
- ``evals.harnesses`` — run eval pipelines (capability, regression)
- ``evals.runner``    — variant experiment CLI + LangFuse integration
- ``evals.utils``     — shared models, loaders, settings, tracing, LangSmith

CLI: ``python -m evals.runner <upload|run> [options]``
"""
