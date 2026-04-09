from __future__ import annotations

import json
from hashlib import sha256

from rag_system.src.rag_core.retrieval.retriever import RetrievalResult, Retriever


class EnsembleRetriever(Retriever):
    """Composite retriever combining multiple retrievers with weights."""

    def __init__(self, retrievers_with_weights: list[tuple[Retriever, float]]) -> None:
        self.retrievers_with_weights = retrievers_with_weights

    def _doc_hash(self, doc: dict) -> str:
        return sha256(json.dumps(doc, sort_keys=True).encode("utf-8")).hexdigest()

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        doc_scores: dict[str, float] = {}
        doc_objects: dict[str, dict] = {}

        for retriever, weight in self.retrievers_with_weights:
            results = retriever.retrieve(query, k=k)
            for rr in results:
                key = self._doc_hash(rr.document)
                doc_scores[key] = doc_scores.get(key, 0.0) + rr.score * weight
                doc_objects[key] = rr.document

        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            RetrievalResult(document=doc_objects[doc_id], score=score)
            for doc_id, score in sorted_items
        ]

    def encode(self, text: str) -> list[float]:
        if self.retrievers_with_weights:
            return self.retrievers_with_weights[0][0].encode(text)
        return []
