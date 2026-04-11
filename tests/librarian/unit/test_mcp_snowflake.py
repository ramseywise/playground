"""Tests for the Snowflake MCP server."""

from __future__ import annotations

import pytest

from interfaces.mcp.snowflake_server import SnowflakeClient, _SQL_PREFIX_RE


class TestSqlValidation:
    """Test the SQL prefix allowlist."""

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM users",
            "  select count(*) from t",
            "SHOW TABLES",
            "DESCRIBE TABLE users",
            "WITH cte AS (SELECT 1) SELECT * FROM cte",
            "EXPLAIN SELECT 1",
        ],
    )
    def test_safe_queries_pass(self, sql: str) -> None:
        assert _SQL_PREFIX_RE.match(sql) is not None

    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO users VALUES (1)",
            "UPDATE users SET name='x'",
            "DELETE FROM users",
            "DROP TABLE users",
            "CREATE TABLE t (id INT)",
            "ALTER TABLE users ADD COLUMN x INT",
            "TRUNCATE TABLE users",
            "GRANT SELECT ON t TO role_x",
        ],
    )
    def test_unsafe_queries_blocked(self, sql: str) -> None:
        assert _SQL_PREFIX_RE.match(sql) is None


class TestSnowflakeClient:
    def test_execute_rejects_unsafe_sql(self) -> None:
        from unittest.mock import MagicMock

        from librarian.config import LibrarySettings

        cfg = MagicMock(spec=LibrarySettings)
        client = SnowflakeClient(cfg)

        with pytest.raises(ValueError, match="Only read-only"):
            client.execute("DROP TABLE users")

    def test_describe_table_rejects_injection(self) -> None:
        from unittest.mock import MagicMock

        from librarian.config import LibrarySettings

        cfg = MagicMock(spec=LibrarySettings)
        client = SnowflakeClient(cfg)

        with pytest.raises(ValueError, match="Invalid table name"):
            client.describe_table("users; DROP TABLE users")

    def test_describe_table_accepts_valid_names(self) -> None:
        """Valid table names should pass validation (execution mocked)."""
        from unittest.mock import MagicMock, patch

        from librarian.config import LibrarySettings

        cfg = MagicMock(spec=LibrarySettings)
        client = SnowflakeClient(cfg)

        with patch.object(client, "execute", return_value=[]) as mock_exec:
            client.describe_table("my_schema.users")
            mock_exec.assert_called_once_with("DESCRIBE TABLE my_schema.users")

    def test_execute_returns_dicts(self) -> None:
        """Verify execute() converts cursor rows to dicts."""
        from unittest.mock import MagicMock

        from librarian.config import LibrarySettings

        cfg = MagicMock(spec=LibrarySettings)
        client = SnowflakeClient(cfg)

        mock_cursor = MagicMock()
        mock_cursor.description = [("id",), ("name",)]
        mock_cursor.fetchall.return_value = [(1, "alice"), (2, "bob")]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        client._conn = mock_conn

        rows = client.execute("SELECT id, name FROM users")
        assert rows == [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
