"""Tests for the S3 document loader."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from librarian.ingestion.s3_loader import S3DocumentLoader


@pytest.fixture()
def mock_s3_client() -> MagicMock:
    client = MagicMock()
    return client


@pytest.fixture()
def loader(mock_s3_client: MagicMock) -> S3DocumentLoader:
    ldr = S3DocumentLoader(bucket="test-bucket", region="us-east-1")
    ldr._client = mock_s3_client
    return ldr


class TestLoadObject:
    def test_markdown_with_frontmatter(self, loader: S3DocumentLoader, mock_s3_client: MagicMock) -> None:
        content = b'---\ntitle: "My Doc"\ntopic: retrieval\n---\n\nBody text here.'
        mock_s3_client.get_object.return_value = {
            "Body": BytesIO(content),
        }

        doc = loader.load_object("raw/my_doc.md")

        assert doc["text"] == "Body text here."
        assert doc["title"] == "My Doc"
        assert doc["topic"] == "retrieval"
        assert doc["source_file"] == "s3://test-bucket/raw/my_doc.md"
        assert doc["source"] == "s3"

    def test_plain_text_no_frontmatter(self, loader: S3DocumentLoader, mock_s3_client: MagicMock) -> None:
        content = b"Just plain text content."
        mock_s3_client.get_object.return_value = {
            "Body": BytesIO(content),
        }

        doc = loader.load_object("raw/notes.txt")

        assert doc["text"] == "Just plain text content."
        assert doc["title"] == "Notes"
        assert doc["source_file"] == "s3://test-bucket/raw/notes.txt"

    def test_title_fallback_from_key(self, loader: S3DocumentLoader, mock_s3_client: MagicMock) -> None:
        content = b"Some content."
        mock_s3_client.get_object.return_value = {
            "Body": BytesIO(content),
        }

        doc = loader.load_object("raw/nested/my_document.md")

        assert doc["title"] == "My Document"


class TestListObjects:
    def test_single_page(self, loader: S3DocumentLoader, mock_s3_client: MagicMock) -> None:
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "raw/a.md"}, {"Key": "raw/b.md"}],
            "IsTruncated": False,
        }

        keys = loader.list_objects("raw/")

        assert keys == ["raw/a.md", "raw/b.md"]

    def test_paginated(self, loader: S3DocumentLoader, mock_s3_client: MagicMock) -> None:
        mock_s3_client.list_objects_v2.side_effect = [
            {
                "Contents": [{"Key": "raw/a.md"}],
                "IsTruncated": True,
                "NextContinuationToken": "token1",
            },
            {
                "Contents": [{"Key": "raw/b.md"}],
                "IsTruncated": False,
            },
        ]

        keys = loader.list_objects("raw/")

        assert keys == ["raw/a.md", "raw/b.md"]
        assert mock_s3_client.list_objects_v2.call_count == 2

    def test_empty_prefix(self, loader: S3DocumentLoader, mock_s3_client: MagicMock) -> None:
        mock_s3_client.list_objects_v2.return_value = {
            "IsTruncated": False,
        }

        keys = loader.list_objects("empty/")

        assert keys == []


class TestLoadPrefix:
    def test_filters_by_extension(self, loader: S3DocumentLoader, mock_s3_client: MagicMock) -> None:
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "raw/doc.md"},
                {"Key": "raw/image.png"},
                {"Key": "raw/page.html"},
            ],
            "IsTruncated": False,
        }
        # Mock get_object for the two valid files
        mock_s3_client.get_object.side_effect = [
            {"Body": BytesIO(b"Doc content")},
            {"Body": BytesIO(b"Page content")},
        ]

        docs = loader.load_prefix("raw/")

        assert len(docs) == 2
        assert docs[0]["text"] == "Doc content"
        assert docs[1]["text"] == "Page content"

    def test_source_file_has_s3_uri(self, loader: S3DocumentLoader, mock_s3_client: MagicMock) -> None:
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "raw/test.md"}],
            "IsTruncated": False,
        }
        mock_s3_client.get_object.return_value = {"Body": BytesIO(b"Content")}

        docs = loader.load_prefix("raw/")

        assert docs[0]["source_file"] == "s3://test-bucket/raw/test.md"


class TestLazyClient:
    @patch("boto3.client")
    def test_creates_client_lazily(self, mock_boto3_client: MagicMock) -> None:
        loader = S3DocumentLoader(bucket="b", region="eu-west-1")
        assert loader._client is None

        loader._get_client()

        mock_boto3_client.assert_called_once_with("s3", region_name="eu-west-1")
        assert loader._client is not None
