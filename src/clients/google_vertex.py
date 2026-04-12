"""Thin wrapper around Google Gemini with Vertex AI Search grounding.

This provides out-of-the-box RAG via Google — Gemini handles generation
while Vertex AI Search handles embedding, indexing, and retrieval.
Used as an A/B comparison baseline alongside Bedrock KB and our custom
Librarian pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from librarian.config import LibrarySettings

log = structlog.get_logger(__name__)


@dataclass
class GoogleRAGResponse:
    """Normalized response from Google Gemini with grounding."""

    response: str
    citations: list[dict[str, str]]
    raw: Any = field(default=None, repr=False)


class GoogleRAGClient:
    """Client for Google Gemini with Vertex AI Search grounding.

    When ``google_project_id`` is set, uses Vertex AI mode with a datastore
    for enterprise grounding.  Otherwise, uses Gemini API with Google Search
    retrieval for web-grounded answers.
    """

    def __init__(self, cfg: LibrarySettings) -> None:
        if not cfg.google_datastore_id and not cfg.google_project_id:
            msg = "google_datastore_id or google_project_id is not configured"
            raise ValueError(msg)
        if cfg.google_datastore_id and not cfg.google_project_id:
            msg = "google_project_id is required when google_datastore_id is set"
            raise ValueError(msg)

        self._cfg = cfg
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai  # type: ignore[import-untyped]

            if self._cfg.google_project_id:
                self._client = genai.Client(
                    vertexai=True,
                    project=self._cfg.google_project_id,
                    location=self._cfg.google_location,
                )
            else:
                self._client = genai.Client(api_key=self._cfg.gemini_api_key)
        return self._client

    def _build_grounding_tool(self) -> Any:
        """Build the appropriate grounding tool based on config."""
        from google.genai import types  # type: ignore[import-untyped]

        if self._cfg.google_datastore_id:
            datastore_path = (
                f"projects/{self._cfg.google_project_id}"
                f"/locations/{self._cfg.google_location}"
                f"/collections/default_collection"
                f"/dataStores/{self._cfg.google_datastore_id}"
            )
            return types.Tool(
                retrieval=types.Retrieval(
                    vertex_ai_search=types.VertexAISearch(datastore=datastore_path),
                ),
            )
        return types.Tool(
            google_search_retrieval=types.GoogleSearchRetrieval(),
        )

    def query(
        self,
        query: str,
        *,
        session_id: str | None = None,
    ) -> GoogleRAGResponse:
        """Send a grounded query to Gemini and return a normalized response."""
        from google.genai import types  # type: ignore[import-untyped]

        client = self._get_client()
        model = self._cfg.model_gemini

        log.info(
            "google_rag.query",
            query=query[:80],
            model=model,
            datastore_id=self._cfg.google_datastore_id,
        )

        config = types.GenerateContentConfig(
            tools=[self._build_grounding_tool()],
        )

        result = client.models.generate_content(
            model=model,
            contents=query,
            config=config,
        )

        response_text = result.text or ""
        citations = _extract_citations(result)

        log.info(
            "google_rag.response",
            response_len=len(response_text),
            citation_count=len(citations),
        )

        return GoogleRAGResponse(
            response=response_text,
            citations=citations,
            raw=result,
        )

    async def aquery(
        self,
        query: str,
        *,
        session_id: str | None = None,
    ) -> GoogleRAGResponse:
        """Async grounded query — uses Gemini's native async API."""
        from google.genai import types  # type: ignore[import-untyped]

        client = self._get_client()
        model = self._cfg.model_gemini

        log.info(
            "google_rag.aquery",
            query=query[:80],
            model=model,
        )

        config = types.GenerateContentConfig(
            tools=[self._build_grounding_tool()],
        )

        result = await client.aio.models.generate_content(
            model=model,
            contents=query,
            config=config,
        )

        response_text = result.text or ""
        citations = _extract_citations(result)

        log.info(
            "google_rag.aresponse",
            response_len=len(response_text),
            citation_count=len(citations),
        )

        return GoogleRAGResponse(
            response=response_text,
            citations=citations,
            raw=result,
        )


def _extract_citations(result: Any) -> list[dict[str, str]]:
    """Extract grounding citations from a Gemini response.

    Gemini returns grounding metadata in ``candidates[0].grounding_metadata``
    with ``grounding_chunks`` containing ``web`` or ``retrieved_context`` items.
    """
    seen: set[str] = set()
    citations: list[dict[str, str]] = []

    try:
        metadata = result.candidates[0].grounding_metadata
        if metadata is None:
            return citations

        for chunk in getattr(metadata, "grounding_chunks", []) or []:
            web = getattr(chunk, "web", None)
            context = getattr(chunk, "retrieved_context", None)

            if web:
                uri = getattr(web, "uri", "") or ""
                title = getattr(web, "title", "") or uri
            elif context:
                uri = getattr(context, "uri", "") or ""
                title = getattr(context, "title", "") or uri
            else:
                continue

            if not uri or uri in seen:
                continue
            seen.add(uri)
            citations.append({"url": uri, "title": str(title)})

    except (IndexError, AttributeError):
        log.debug("google_rag.citations.parse_failed", exc_info=True)

    return citations
