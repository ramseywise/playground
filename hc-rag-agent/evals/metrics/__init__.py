"""Per-stage RAG evaluation metrics.

Submodules:
- ``retrieval``: hit_rate, MRR, precision, recall, NDCG
- ``reranker``: rank displacement, NDCG improvement, score correlation
- ``confidence``: gate accuracy, FPR, FNR, threshold calibration

Import individual submodules directly to avoid pulling in heavy
dependencies at package level::

    from evals.metrics.retrieval import evaluate_retrieval
    from evals.metrics.reranker import evaluate_reranker
    from evals.metrics.confidence import evaluate_gate
"""
