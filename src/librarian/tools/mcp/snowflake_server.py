"""MCP server exposing Snowflake as a read-only tool."""

from __future__ import annotations

import re
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from agents.librarian.utils.config import LibrarySettings, settings as _default_settings
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)

# Only allow read-only SQL prefixes
_SAFE_PREFIXES = ("SELECT", "SHOW", "DESCRIBE", "WITH", "EXPLAIN")
_SQL_PREFIX_RE = re.compile(
    r"^\s*(" + "|".join(_SAFE_PREFIXES) + r")\b", re.IGNORECASE
)


class SnowflakeClient:
    """Thin wrapper around snowflake-connector-python."""

    def __init__(self, cfg: LibrarySettings) -> None:
        self._cfg = cfg
        self._conn: Any = None

    def _get_conn(self) -> Any:
        if self._conn is None:
            import snowflake.connector

            self._conn = snowflake.connector.connect(
                account=self._cfg.snowflake_account,
                user=self._cfg.snowflake_user,
                password=self._cfg.snowflake_password,
                warehouse=self._cfg.snowflake_warehouse,
                database=self._cfg.snowflake_database,
                schema=self._cfg.snowflake_schema,
            )
            log.info("snowflake.connected", account=self._cfg.snowflake_account)
        return self._conn

    def execute(self, sql: str) -> list[dict[str, Any]]:
        """Execute a read-only SQL query and return rows as dicts."""
        if not _SQL_PREFIX_RE.match(sql):
            msg = f"Only read-only queries are allowed. Got: {sql[:60]}"
            raise ValueError(msg)

        conn = self._get_conn()
        cur = conn.cursor()
        try:
            cur.execute(sql)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
        finally:
            cur.close()

    def list_tables(self) -> list[str]:
        """Return table names in the configured schema."""
        rows = self.execute("SHOW TABLES")
        return [r.get("name", r.get("TABLE_NAME", "")) for r in rows]

    def describe_table(self, table: str) -> list[dict[str, Any]]:
        """Return column metadata for a table."""
        # Validate table name to prevent injection
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", table):
            msg = f"Invalid table name: {table}"
            raise ValueError(msg)
        return self.execute(f"DESCRIBE TABLE {table}")  # noqa: S608


def create_server(cfg: LibrarySettings | None = None) -> Server:
    """Create the Snowflake MCP server."""
    settings = cfg or _default_settings
    client = SnowflakeClient(settings)
    server = Server("mcp-snowflake")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="query_snowflake",
                description="Execute a read-only SQL query against Snowflake. Only SELECT, SHOW, DESCRIBE, WITH, and EXPLAIN are allowed.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "SQL query to execute"},
                    },
                    "required": ["sql"],
                },
            ),
            Tool(
                name="list_tables",
                description="List all tables in the configured Snowflake schema.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="describe_table",
                description="Get column schema for a Snowflake table.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "description": "Table name"},
                    },
                    "required": ["table"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        import json

        if name == "query_snowflake":
            rows = client.execute(arguments["sql"])
            return [TextContent(type="text", text=json.dumps(rows, default=str))]

        if name == "list_tables":
            tables = client.list_tables()
            return [TextContent(type="text", text=json.dumps(tables))]

        if name == "describe_table":
            schema = client.describe_table(arguments["table"])
            return [TextContent(type="text", text=json.dumps(schema, default=str))]

        msg = f"Unknown tool: {name}"
        raise ValueError(msg)

    return server


def main() -> None:
    """Entry point for ``mcp-snowflake`` console script."""
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
