from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Matches a YAML-style frontmatter block at the top of a Markdown file.
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# Matches a single frontmatter key: value line (unquoted or double-quoted value).
_KV_RE = re.compile(r'^(\w[\w_-]*)\s*:\s*"?([^"\n]*)"?\s*$', re.MULTILINE)


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Extract YAML-lite frontmatter and return (metadata_dict, body_text)."""
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw

    fm_block = match.group(1)
    body = raw[match.end() :]
    meta: dict[str, str] = {}
    for key, value in _KV_RE.findall(fm_block):
        cleaned = value.strip()
        if cleaned.lower() == "null":
            cleaned = ""
        meta[key] = cleaned
    return meta, body


def load_markdown_file(path: Path) -> dict[str, str]:
    """Load a single Markdown file and return a doc dict.

    Returns keys: text, title, url, source, content_type, topic, source_file.
    Frontmatter values override defaults; missing keys fall back to empty strings.
    """
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)

    doc = {
        "text": body.strip(),
        "title": meta.get("title", path.stem.replace("_", " ").title()),
        "url": meta.get("url", ""),
        "source": meta.get("source", "blog"),
        "content_type": meta.get("content_type", "article"),
        "topic": meta.get("topic", ""),
        "source_file": str(path),
    }
    log.debug("loader.markdown.loaded path=%s title=%s", path, doc["title"])
    return doc


def load_directory(
    directory: Path,
    glob_pattern: str = "*.md",
) -> list[dict[str, str]]:
    """Load all matching files from *directory* and return a sorted list of docs."""
    paths = sorted(directory.glob(glob_pattern))
    docs = [load_markdown_file(p) for p in paths]
    log.info("loader.directory.loaded directory=%s count=%d", directory, len(docs))
    return docs


_SUPPORTED_S3_EXTENSIONS = (".md", ".html", ".txt")


class S3DocumentLoader:
    """Load documents from S3 using the same dict contract as local loaders.

    Returns ``dict[str, str]`` with keys: text, title, url, source,
    content_type, topic, source_file. ``source_file`` is set to
    ``s3://{bucket}/{key}`` so checksums differentiate S3 vs local paths.
    """

    def __init__(self, bucket: str, region: str = "") -> None:
        self._bucket = bucket
        self._region = region
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy boto3 S3 client."""
        if self._client is None:
            import boto3  # type: ignore[import-untyped]

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

        meta, body = parse_frontmatter(raw)
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
        log.debug(
            "s3_loader.loaded bucket=%s key=%s title=%s",
            self._bucket,
            key,
            doc["title"],
        )
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
        extensions: tuple[str, ...] = _SUPPORTED_S3_EXTENSIONS,
    ) -> list[dict[str, str]]:
        """Load all matching objects under *prefix*."""
        keys = self.list_objects(prefix)
        docs: list[dict[str, str]] = []
        for key in sorted(keys):
            if any(key.endswith(ext) for ext in extensions):
                docs.append(self.load_object(key))
            else:
                log.debug("s3_loader.skip_extension key=%s", key)
        log.info(
            "s3_loader.prefix.loaded bucket=%s prefix=%s count=%d",
            self._bucket,
            prefix,
            len(docs),
        )
        return docs
