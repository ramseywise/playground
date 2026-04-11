"""MCP server exposing the Librarian RAG agent as a tool."""

from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from librarian.config import LibrarySettings, settings as _default_settings
from core.logging import get_logger

log = get_logger(__name__)

# Lazy singletons — initialised on first tool call
_graph: Any = None
_pipeline: Any = None
_settings: LibrarySettings = _default_settings


def _get_graph(cfg: LibrarySettings) -> Any:
    global _graph  # noqa: PLW0603
    if _graph is None:
        from librarian.factory import create_librarian

        _graph = create_librarian(cfg)
        log.info("librarian_mcp.graph.init")
    return _graph


def _get_pipeline(cfg: LibrarySettings) -> Any:
    global _pipeline  # noqa: PLW0603
    if _pipeline is None:
        from librarian.factory import create_ingestion_pipeline

        _pipeline = create_ingestion_pipeline(cfg)
        log.info("librarian_mcp.pipeline.init")
    return _pipeline


def create_server(cfg: LibrarySettings | None = None) -> Server:
    """Create the Librarian MCP server."""
    settings = cfg or _default_settings
    server = Server("mcp-librarian")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="search",
                description="Search the knowledge base and return ranked chunks with relevance scores.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="chat",
                description="Ask the Librarian a question and get a sourced answer with citations.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Question to ask"},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="ingest",
                description="Ingest a document into the knowledge base. Provide a JSON string with at least a 'text' field.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document": {"type": "string", "description": "JSON string of the document dict (must include 'text' key)"},
                    },
                    "required": ["document"],
                },
            ),
            Tool(
                name="get_status",
                description="Get knowledge base metadata and configuration info.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "search":
            graph = _get_graph(settings)
            result = await graph.ainvoke({"query": arguments["query"]})
            chunks = result.get("reranked_chunks") or result.get("retrieved_chunks") or []
            return [TextContent(type="text", text=json.dumps(chunks, default=str))]

        if name == "chat":
            graph = _get_graph(settings)
            result = await graph.ainvoke({"query": arguments["query"]})
            return [TextContent(
                type="text",
                text=json.dumps({
                    "response": result.get("response", ""),
                    "citations": result.get("citations", []),
                    "confidence_score": result.get("confidence_score", 0.0),
                    "intent": result.get("intent", ""),
                }, default=str),
            )]

        if name == "ingest":
            pipeline = _get_pipeline(settings)
            doc = json.loads(arguments["document"])
            r = await pipeline.ingest_document(doc)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "doc_id": r.doc_id,
                    "chunk_count": r.chunk_count,
                    "snippet_count": r.snippet_count,
                    "skipped": r.skipped,
                }),
            )]

        if name == "get_status":
            return [TextContent(
                type="text",
                text=json.dumps({
                    "model": settings.anthropic_model_sonnet,
                    "retrieval_strategy": settings.retrieval_strategy,
                    "confidence_threshold": settings.confidence_threshold,
                    "s3_bucket": settings.s3_bucket,
                }),
            )]

        msg = f"Unknown tool: {name}"
        raise ValueError(msg)

    return server


def main() -> None:
    """Entry point for ``mcp-librarian`` console script."""
    import asyncio

    async def _run() -> None:
        server = create_server()
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options(),
            )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
