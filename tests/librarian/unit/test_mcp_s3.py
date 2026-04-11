"""Tests for the S3 MCP server."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from interfaces.mcp.s3_server import S3Client


@pytest.fixture()
def cfg() -> MagicMock:
    mock = MagicMock()
    mock.s3_bucket = "test-bucket"
    mock.s3_region = "us-east-1"
    mock.s3_raw_prefix = "raw/"
    return mock


class TestS3Client:
    def test_list_objects(self, cfg: MagicMock) -> None:
        client = S3Client(cfg)
        mock_boto = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "raw/a.md", "Size": 100}, {"Key": "raw/b.md", "Size": 200}]}
        ]
        mock_boto.get_paginator.return_value = paginator
        client._client = mock_boto

        result = client.list_objects("raw/")
        assert len(result) == 2
        assert result[0] == {"key": "raw/a.md", "size": 100}

    def test_get_object(self, cfg: MagicMock) -> None:
        client = S3Client(cfg)
        mock_boto = MagicMock()
        mock_boto.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=b"hello world"))
        }
        client._client = mock_boto

        content = client.get_object("raw/doc.md")
        assert content == "hello world"

    def test_put_object_auto_prefixes(self, cfg: MagicMock) -> None:
        client = S3Client(cfg)
        mock_boto = MagicMock()
        client._client = mock_boto

        client.put_object("doc.md", "content")
        mock_boto.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="raw/doc.md",
            Body=b"content",
        )

    def test_put_object_no_double_prefix(self, cfg: MagicMock) -> None:
        client = S3Client(cfg)
        mock_boto = MagicMock()
        client._client = mock_boto

        client.put_object("raw/doc.md", "content")
        mock_boto.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="raw/doc.md",
            Body=b"content",
        )

    def test_lazy_client_init(self, cfg: MagicMock) -> None:
        client = S3Client(cfg)
        assert client._client is None
