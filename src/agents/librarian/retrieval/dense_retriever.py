from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from rag_system.src.rag_core.retrieval.retriever import RetrievalResult, Retriever
from rag_system.src.rag_core.utils.logging import get_logger

log = get_logger(__name__)


class DenseRetriever(Retriever, ABC):
    """Abstract base class for dense retrievers using neural embeddings."""

    def __init__(self, embedding_store, metadata_store=None, chunker=None) -> None:
        self.embedding_store = embedding_store
        self.metadata_store = metadata_store
        self.chunker = chunker
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: PreTrainedModel | None = None

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            tokenizer, model = self.load_model()
            self._tokenizer = tokenizer
            self._model = model.eval()
        assert self._tokenizer is not None
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            tokenizer, model = self.load_model()
            self._tokenizer = tokenizer
            self._model = model.eval()
        assert self._model is not None
        return self._model

    @abstractmethod
    def load_model(self):
        """Return (tokenizer, model) tuple."""
        raise NotImplementedError

    def encode(self, text: str) -> list[float]:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            output = self.model(**inputs)
        return output.last_hidden_state.mean(dim=1).squeeze().tolist()

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        query_vector = self.encode(query)
        id_score_pairs = self.embedding_store.search(k, query_vector)

        results: list[RetrievalResult] = []
        for chunk_id, score in id_score_pairs:
            try:
                doc = self.embedding_store.get_document(chunk_id)
            except Exception as exc:
                log.error(
                    "dense_retriever.fetch_failed", chunk_id=chunk_id, error=str(exc)
                )
                continue
            results.append(RetrievalResult(document=doc, score=score))
        return results


class SentenceTransformerRetriever(DenseRetriever):
    """DenseRetriever using a SentenceTransformer model."""

    def __init__(
        self,
        embedding_store,
        metadata_store=None,
        sen_trans_model: str | SentenceTransformer = "all-MiniLM-L6-v2",
        chunker=None,
    ) -> None:
        if isinstance(sen_trans_model, str):
            self.sen_trans = SentenceTransformer(sen_trans_model)
        elif isinstance(sen_trans_model, SentenceTransformer):
            self.sen_trans = sen_trans_model
        else:
            raise ValueError(
                f"sen_trans_model must be str or SentenceTransformer, got {type(sen_trans_model)}"
            )
        super().__init__(embedding_store, metadata_store, chunker=chunker)

    def load_model(self):
        return None, None

    def encode(self, text: str) -> list[float]:
        import inspect

        sig = inspect.signature(self.sen_trans.encode)
        supported = sig.parameters.keys()

        encode_kwargs: dict = {}
        if "convert_to_numpy" in supported:
            encode_kwargs["convert_to_numpy"] = True
        if "normalize_embeddings" in supported:
            encode_kwargs["normalize_embeddings"] = True

        embs = self.sen_trans.encode([text], **encode_kwargs)

        if isinstance(embs, torch.Tensor):
            embs = embs.cpu().numpy()
        elif (
            isinstance(embs, list)
            and len(embs) > 0
            and isinstance(embs[0], torch.Tensor)
        ):
            embs = np.array([t.cpu().numpy() for t in embs])

        np_embs = np.array(embs)
        vec = np_embs[0]

        if getattr(self, "normalize_embeddings", False):
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        return vec.tolist()
