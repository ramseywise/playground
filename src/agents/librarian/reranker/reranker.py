from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from hashlib import sha256

from rag_system.src.rag_core.retrieval.retriever import RetrievalResult
from rag_system.src.rag_core.utils.logging import get_logger

log = get_logger(__name__)

RERANKER_MAX_INPUT_CHARS: int = 2000


class Reranker(ABC):
    """Abstract base class for re-rankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Re-rank retrieval results by relevance to query."""
        pass


class CrossEncoderReranker(Reranker):
    """Re-ranker using cross-encoder models for better relevance scoring."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = None
        self._device = device

    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, device=self._device)
                log.info("reranker.cross_encoder.loaded", model=self.model_name)
            except ImportError:
                log.error("reranker.cross_encoder.import_error")
                raise
        return self._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        texts = []
        for r in results:
            doc = r.document
            text = doc.get("Text") or doc.get("text") or doc.get("content", "")
            texts.append(text[:RERANKER_MAX_INPUT_CHARS])

        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs)

        scored = [
            RetrievalResult(document=r.document, score=float(s))
            for r, s in zip(results, scores)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)

        if top_k is not None:
            scored = scored[:top_k]
        return scored


class RRFReranker(Reranker):
    """Re-ranker using Reciprocal Rank Fusion."""

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        rrf_results = [
            RetrievalResult(document=r.document, score=1.0 / (self.k + rank))
            for rank, r in enumerate(results, 1)
        ]

        if top_k is not None:
            rrf_results = rrf_results[:top_k]
        return rrf_results

    def fuse_rankings(
        self,
        rankings: list[list[RetrievalResult]],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Fuse multiple ranked lists using RRF."""
        doc_scores: dict[str, float] = {}
        doc_objects: dict[str, dict] = {}

        for ranking in rankings:
            for rank, r in enumerate(ranking, 1):
                doc_hash = sha256(
                    json.dumps(r.document, sort_keys=True).encode()
                ).hexdigest()
                rrf_score = 1.0 / (self.k + rank)
                doc_scores[doc_hash] = doc_scores.get(doc_hash, 0.0) + rrf_score
                doc_objects[doc_hash] = r.document

        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        results = [
            RetrievalResult(document=doc_objects[h], score=s) for h, s in sorted_items
        ]

        if top_k is not None:
            results = results[:top_k]
        return results


class LLMReranker(Reranker):
    """Re-ranker using LLM to score relevance."""

    def __init__(self, llm_caller, batch_size: int = 5) -> None:
        self.llm_caller = llm_caller
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        scored_results = []
        for r in results:
            doc = r.document
            text = doc.get("Text") or doc.get("text") or doc.get("content", "")

            prompt = f"""Rate the relevance of this document to the query on a scale of 1-10.

Query: {query}

Document: {text[:500]}

Respond with only a JSON object: {{"score": <1-10>, "reason": "<brief reason>"}}"""

            try:
                response = self.llm_caller(prompt)
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                    score = float(result.get("score", 5)) / 10.0
                else:
                    score = r.score
            except Exception:
                score = r.score

            scored_results.append(RetrievalResult(document=r.document, score=score))

        scored_results.sort(key=lambda x: x.score, reverse=True)

        if top_k is not None:
            scored_results = scored_results[:top_k]
        return scored_results


class ColBERTReranker(Reranker):
    """Re-ranker using ColBERT late interaction model.

    Requires: pip install ragatouille or colbert-ai
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        device: str | None = None,
        use_ragatouille: bool = True,
    ) -> None:
        self.model_name = model_name
        self._model = None
        self._device = device
        self.use_ragatouille = use_ragatouille

    @property
    def model(self):
        """Lazy-load the ColBERT model."""
        if self._model is None:
            if self.use_ragatouille:
                try:
                    from ragatouille import RAGPretrainedModel

                    self._model = RAGPretrainedModel.from_pretrained(self.model_name)
                    log.info(
                        "reranker.colbert.loaded",
                        model=self.model_name,
                        backend="ragatouille",
                    )
                except ImportError:
                    log.warning(
                        "reranker.colbert.ragatouille_missing", fallback="direct"
                    )
                    self.use_ragatouille = False

            if not self.use_ragatouille:
                try:
                    from colbert.infra import ColBERTConfig
                    from colbert.modeling.checkpoint import Checkpoint

                    config = ColBERTConfig(checkpoint=self.model_name)
                    self._model = Checkpoint(self.model_name, colbert_config=config)
                    log.info(
                        "reranker.colbert.loaded",
                        model=self.model_name,
                        backend="direct",
                    )
                except ImportError as exc:
                    raise ImportError(
                        "Install ColBERT support: pip install ragatouille or pip install colbert-ai"
                    ) from exc
        return self._model

    def _compute_maxsim(self, query_embs, doc_embs) -> float:
        import torch

        similarities = torch.matmul(query_embs, doc_embs.T)
        return similarities.max(dim=1).values.sum().item()

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        texts = []
        for r in results:
            doc = r.document
            text = doc.get("Text") or doc.get("text") or doc.get("content", "")
            texts.append(text[:RERANKER_MAX_INPUT_CHARS])

        if self.use_ragatouille:
            try:
                reranked = self.model.rerank(
                    query=query, documents=texts, k=top_k or len(texts)
                )
                scored_results = []
                for item in reranked:
                    if isinstance(item, dict):
                        idx = item.get("result_index", item.get("idx", 0))
                        score = item.get("score", 0.0)
                    else:
                        idx = texts.index(item[0]) if isinstance(item[0], str) else 0
                        score = item[1] if len(item) > 1 else 0.0
                    if idx < len(results):
                        scored_results.append(
                            RetrievalResult(
                                document=results[idx].document, score=float(score)
                            )
                        )
                return scored_results[:top_k] if top_k else scored_results
            except Exception as exc:
                log.warning("reranker.colbert.rerank_failed", error=str(exc))
                return results[:top_k] if top_k else results
        else:
            import torch

            try:
                query_embs = self.model.queryFromText([query])[0]
                scored_results = []
                for r, text in zip(results, texts):
                    doc_embs = self.model.docFromText([text])[0]
                    score = self._compute_maxsim(
                        torch.tensor(query_embs), torch.tensor(doc_embs)
                    )
                    scored_results.append(
                        RetrievalResult(document=r.document, score=float(score))
                    )
                scored_results.sort(key=lambda x: x.score, reverse=True)
                if top_k is not None:
                    scored_results = scored_results[:top_k]
                return scored_results
            except Exception as exc:
                log.warning("reranker.colbert.scoring_failed", error=str(exc))
                return results[:top_k] if top_k else results
