"""Load documents from S3 for ingestion into the Librarian pipeline."""

from __future__ import annotations

from typing import Any

from librarian.ingestion.loaders import _parse_frontmatter
from core.logging import get_logger

log = get_logger(__name__)

_SUPPORTED_EXTENSIONS = (".md", ".html", ".txt")


class S3DocumentLoader:
    """Load documents from S3 following the same dict contract as local loaders.

    Returns ``dict[str, str]`` with keys: text, title, url, source,
    content_type, topic, source_file.  ``source_file`` is set to
    ``s3://{bucket}/{key}`` so checksums differentiate S3 vs local paths.
    """

    def __init__(self, bucket: str, region: str = "") -> None:
        self._bucket = bucket
        self._region = region
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy boto3 S3 client."""
        if self._client is None:
            import boto3

            kwargs: dict[str, str] = {}
            if self._region:
                kwargs["region_name"] = self._region
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def load_object(self, key: str) -> dict[str, str]:
        """Load a single S3 object and return a doc dict."""
        client = self._get_client()
        response = client.get_object(Bucket=self._bucket, Key=key)
        raw = response["Body"].read().decode("utf-8")

        meta, body = _parse_frontmatter(raw)
        stem = key.rsplit("/", 1)[-1].rsplit(".", 1)[0]

        doc = {
            "text": body.strip(),
            "title": meta.get("title", stem.replace("_", " ").title()),
            "url": meta.get("url", ""),
            "source": meta.get("source", "s3"),
            "content_type": meta.get("content_type", "article"),
            "topic": meta.get("topic", ""),
            "source_file": f"s3://{self._bucket}/{key}",
        }
        log.debug("s3_loader.loaded", bucket=self._bucket, key=key, title=doc["title"])
        return doc

    def list_objects(self, prefix: str) -> list[str]:
        """List all object keys under *prefix*, handling pagination."""
        client = self._get_client()
        keys: list[str] = []
        continuation_token: str | None = None

        while True:
            kwargs: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token

            response = client.list_objects_v2(**kwargs)
            for obj in response.get("Contents", []):
                keys.append(obj["Key"])

            if response.get("IsTruncated"):
                continuation_token = response["NextContinuationToken"]
            else:
                break

        return keys

    def load_prefix(
        self,
        prefix: str,
        extensions: tuple[str, ...] = _SUPPORTED_EXTENSIONS,
    ) -> list[dict[str, str]]:
        """Load all matching objects under *prefix*."""
        keys = self.list_objects(prefix)
        docs: list[dict[str, str]] = []
        for key in sorted(keys):
            if any(key.endswith(ext) for ext in extensions):
                docs.append(self.load_object(key))
            else:
                log.debug("s3_loader.skip_extension", key=key)
        log.info(
            "s3_loader.prefix.loaded",
            bucket=self._bucket,
            prefix=prefix,
            count=len(docs),
        )
        return docs
