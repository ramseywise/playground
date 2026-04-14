"""Thin wrapper around AWS Bedrock Knowledge Bases RetrieveAndGenerate.

This provides out-of-the-box RAG — AWS handles embedding, vector storage,
retrieval, and generation.  Used as an A/B comparison baseline against our
custom Librarian pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import boto3
import structlog

from librarian.config import LibrarySettings

log = structlog.get_logger(__name__)


@dataclass
class BedrockKBResponse:
    """Normalized response from Bedrock Knowledge Bases."""

    response: str
    citations: list[dict[str, str]]
    session_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class BedrockKBClient:
    """Client for AWS Bedrock Knowledge Bases RetrieveAndGenerate API."""

    def __init__(self, cfg: LibrarySettings) -> None:
        if not cfg.bedrock_knowledge_base_id:
            msg = "bedrock_knowledge_base_id is not configured"
            raise ValueError(msg)

        region = cfg.bedrock_region or cfg.s3_region or None
        self._client = boto3.client(
            "bedrock-agent-runtime",
            **({"region_name": region} if region else {}),
        )
        self._kb_id = cfg.bedrock_knowledge_base_id
        self._model_arn = cfg.bedrock_model_arn

    def query(
        self,
        query: str,
        *,
        session_id: str | None = None,
    ) -> BedrockKBResponse:
        """Send a query to Bedrock Knowledge Bases and return a normalized response."""
        params: dict[str, Any] = {
            "input": {"text": query},
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": self._kb_id,
                    "modelArn": self._model_arn,
                },
            },
        }
        if session_id:
            params["sessionId"] = session_id

        log.info(
            "bedrock_kb.query",
            query=query[:80],
            kb_id=self._kb_id,
            session_id=session_id,
        )

        result = self._client.retrieve_and_generate(**params)

        response_text = result.get("output", {}).get("text", "")
        citations = _extract_citations(result.get("citations", []))
        bedrock_session_id = result.get("sessionId", "")

        log.info(
            "bedrock_kb.response",
            response_len=len(response_text),
            citation_count=len(citations),
            session_id=bedrock_session_id,
        )

        return BedrockKBResponse(
            response=response_text,
            citations=citations,
            session_id=bedrock_session_id,
            raw=result,
        )

    async def aquery(
        self,
        query: str,
        *,
        session_id: str | None = None,
    ) -> BedrockKBResponse:
        """Async wrapper — boto3 is sync, so this runs in a thread executor."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.query(query, session_id=session_id)
        )


def _extract_citations(raw_citations: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Flatten Bedrock's nested citation structure to [{url, title}]."""
    seen: set[str] = set()
    citations: list[dict[str, str]] = []

    for citation in raw_citations:
        for ref in citation.get("retrievedReferences", []):
            location = ref.get("location", {})
            uri = (
                location.get("s3Location", {}).get("uri", "")
                or location.get("webLocation", {}).get("url", "")
            )
            if not uri or uri in seen:
                continue
            seen.add(uri)

            metadata = ref.get("metadata", {})
            title = (
                metadata.get("x-amz-bedrock-kb-source-uri-title")
                or metadata.get("title")
                or uri.rsplit("/", 1)[-1]
                or uri
            )

            citations.append({"url": uri, "title": str(title)})

    return citations
