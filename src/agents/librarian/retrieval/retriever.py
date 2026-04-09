"""Retriever for the RAG system."""

from abc import ABC, abstractmethod
from typing import NamedTuple


class RetrievalResult(NamedTuple):
    """Return score value for retrieved document."""

    document: dict
    score: float


class Retriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query (str): The query string.
            k (int): Number of top documents to retrieve.

        Returns:
            List of RetrievalResult(document: Dict, score: float)

        """
        pass

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        """Encode input text to a list of floats.

        Args:
            text (str): Input text.

        Returns:
            List[float]: The text embedding.

        """
        pass
