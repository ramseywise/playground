"""MCP server exposing S3 document management tools."""

from __future__ import annotations

from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from agents.librarian.utils.config import LibrarySettings, settings as _default_settings
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)


class S3Client:
    """Thin wrapper around boto3 S3 for MCP tools."""

    def __init__(self, cfg: LibrarySettings) -> None:
        self._cfg = cfg
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3

            kwargs: dict[str, str] = {}
            if self._cfg.s3_region:
                kwargs["region_name"] = self._cfg.s3_region
            self._client = boto3.client("s3", **kwargs)
            log.info("s3_mcp.connected", bucket=self._cfg.s3_bucket)
        return self._client

    def list_objects(self, prefix: str) -> list[dict[str, Any]]:
        """List objects under a prefix, returning key and size."""
        client = self._get_client()
        paginator = client.get_paginator("list_objects_v2")
        results: list[dict[str, Any]] = []
        for page in paginator.paginate(Bucket=self._cfg.s3_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                results.append({"key": obj["Key"], "size": obj["Size"]})
        return results

    def get_object(self, key: str) -> str:
        """Read an object's content as UTF-8 text."""
        client = self._get_client()
        resp = client.get_object(Bucket=self._cfg.s3_bucket, Key=key)
        return resp["Body"].read().decode("utf-8")

    def put_object(self, key: str, content: str) -> None:
        """Write text content to an S3 object under the raw prefix."""
        if not key.startswith(self._cfg.s3_raw_prefix):
            key = f"{self._cfg.s3_raw_prefix}{key}"
        client = self._get_client()
        client.put_object(
            Bucket=self._cfg.s3_bucket, Key=key, Body=content.encode("utf-8"),
        )
        log.info("s3_mcp.put", key=key)


def create_server(cfg: LibrarySettings | None = None) -> Server:
    """Create the S3 MCP server."""
    settings = cfg or _default_settings
    client = S3Client(settings)
    server = Server("mcp-s3")

    _pipeline: Any = None

    def _get_pipeline() -> Any:
        nonlocal _pipeline
        if _pipeline is None:
            from agents.librarian.factory import create_ingestion_pipeline

            _pipeline = create_ingestion_pipeline(settings)
            log.info("s3_mcp.pipeline.init")
        return _pipeline

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="list_documents",
                description="List documents in the S3 data lake under a given prefix.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prefix": {
                            "type": "string",
                            "description": "S3 prefix to list (default: raw/)",
                            "default": "raw/",
                        },
                    },
                },
            ),
            Tool(
                name="get_document",
                description="Read the content of an S3 document by key.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "S3 object key"},
                    },
                    "required": ["key"],
                },
            ),
            Tool(
                name="upload_document",
                description="Upload a text document to the raw/ prefix for ingestion.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Object key (auto-prefixed with raw/ if needed)"},
                        "content": {"type": "string", "description": "Document content"},
                    },
                    "required": ["key", "content"],
                },
            ),
            Tool(
                name="trigger_ingestion",
                description="Ingest an S3 object into the knowledge base.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "S3 object key to ingest"},
                    },
                    "required": ["key"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        import json

        if name == "list_documents":
            prefix = arguments.get("prefix", settings.s3_raw_prefix)
            objects = client.list_objects(prefix)
            return [TextContent(type="text", text=json.dumps(objects, default=str))]

        if name == "get_document":
            content = client.get_object(arguments["key"])
            return [TextContent(type="text", text=content)]

        if name == "upload_document":
            client.put_object(arguments["key"], arguments["content"])
            return [TextContent(type="text", text=json.dumps({"status": "uploaded", "key": arguments["key"]}))]

        if name == "trigger_ingestion":
            pipeline = _get_pipeline()
            result = await pipeline.ingest_s3_object(
                bucket=settings.s3_bucket,
                key=arguments["key"],
                region=settings.s3_region,
            )
            return [TextContent(
                type="text",
                text=json.dumps({
                    "doc_id": result.doc_id,
                    "chunk_count": result.chunk_count,
                    "snippet_count": result.snippet_count,
                    "skipped": result.skipped,
                }),
            )]

        msg = f"Unknown tool: {name}"
        raise ValueError(msg)

    return server


def main() -> None:
    """Entry point for ``mcp-s3`` console script."""
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
