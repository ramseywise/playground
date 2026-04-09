from __future__ import annotations

from abc import ABC, abstractmethod

import bm25s
import Stemmer

from rag_system.src.rag_core.retrieval.retriever import RetrievalResult, Retriever
from rag_system.src.rag_core.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_STEMMER_LANG = "english"


class SparseRetriever(Retriever, ABC):
    """Abstract base for sparse retrievers."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        pass

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        pass


class BM25Retriever(SparseRetriever):
    """Retriever using BM25 algorithm via bm25s."""

    def __init__(
        self,
        index_path: str,
        corpus: list[dict],
        stemmer_lang: str = DEFAULT_STEMMER_LANG,
    ) -> None:
        """Initialize the BM25 retriever.

        Args:
            index_path: File path to the saved BM25 index.
            corpus: List of document dicts with 'text'/'Text' and 'url'/'URL' keys.
            stemmer_lang: PyStemmer language (default "english").
        """
        self.model = bm25s.BM25.load(index_path)
        self.corpus = corpus
        self.stemmer_lang = stemmer_lang

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        stemmer = Stemmer.Stemmer(self.stemmer_lang)
        tokens = bm25s.tokenize(query, stemmer=stemmer)
        indices, scores = self.model.retrieve(tokens, k=k)
        top_idxs = indices[0]
        top_scores = scores[0]
        return [
            RetrievalResult(document=self.corpus[i], score=float(top_scores[idx]))
            for idx, i in enumerate(top_idxs)
        ]

    def encode(self, text: str) -> list[float]:
        raise NotImplementedError("BM25Retriever does not support encode()")

    @property
    def embedding_dimension(self) -> int:
        raise NotImplementedError("BM25 has no fixed embedding dimension")
