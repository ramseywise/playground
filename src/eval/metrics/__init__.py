"""Per-stage RAG evaluation metrics.

Submodules:
- ``retrieval``: hit_rate, MRR, precision, recall, NDCG
- ``reranker``: rank displacement, NDCG improvement, score correlation
- ``confidence``: gate accuracy, FPR, FNR, threshold calibration

Import individual submodules directly to avoid pulling in heavy
dependencies (e.g. ``core.logging``, ``librarian``) at package level::

    from eval.metrics.retrieval import evaluate_retrieval
    from eval.metrics.reranker import evaluate_reranker
    from eval.metrics.confidence import evaluate_gate
"""
