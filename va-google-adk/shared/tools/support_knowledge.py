"""Support knowledge tool — queries the Billy help documentation.

Production: calls AWS Bedrock Knowledge Base.
Development: set SUPPORT_STUB=1 to return a canned stub response.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "4SUAFKZBE8")
_AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
_AWS_PROFILE = os.getenv("AWS_PROFILE")
_SCORE_THRESHOLD = 0.4
_NUM_RESULTS = 5
_STUB = os.getenv("SUPPORT_STUB", "0") == "1"

_STUB_RESPONSE = (
    "[PASSAGE 1] score=0.85 | https://help.billy.dk/creating-invoices\n"
    "URL: https://help.billy.dk/creating-invoices\n"
    "To create an invoice in Billy, go to Sales → Invoices → Create invoice. "
    "Select the customer, add at least one product line item, and click Approve."
)


def _get_bedrock_client():
    import boto3
    session = boto3.Session(profile_name=_AWS_PROFILE, region_name=_AWS_REGION)
    return session.client("bedrock-agent-runtime")


def _extract_url(location: dict[str, Any]) -> str | None:
    for key in ("webLocation", "s3Location", "confluenceLocation", "sharePointLocation"):
        if key in location:
            return location[key].get("url") or location[key].get("uri")
    return None


def _format_passages(results: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, result in enumerate(results, start=1):
        score: float = result.get("score", 0.0)
        if score < _SCORE_THRESHOLD:
            continue
        text = result.get("content", {}).get("text", "").strip()
        if not text:
            continue
        location = result.get("location", {})
        url = _extract_url(location)
        metadata = result.get("metadata", {})
        title = metadata.get("title") or url
        header = f"[PASSAGE {i}] score={score:.2f}"
        if title:
            header += f" | {title}"
        if url:
            header += f"\nURL: {url}"
        parts.append(f"{header}\n{text}")
    return "\n---\n".join(parts)


def _deduplicate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
    seen: set[str] = set()
    unique: list[dict] = []
    for r in sorted_results:
        url = _extract_url(r.get("location", {})) or "no-url"
        snippet = "".join(r.get("content", {}).get("text", "")[:200].split()).lower()
        fp = f"{url}|{snippet}"
        if fp not in seen:
            unique.append(r)
            seen.add(fp)
    return unique


def _retrieve_raw(query: str) -> list[dict]:
    client = _get_bedrock_client()
    response = client.retrieve(
        knowledgeBaseId=_KB_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": _NUM_RESULTS,
                "overrideSearchType": "HYBRID",
            }
        },
    )
    return response.get("retrievalResults", [])


async def fetch_support_knowledge(queries: list[str]) -> str:
    """Search the official Billy support documentation.

    Args:
        queries: 2-3 search terms or phrases (Danish preferred).
                 Example: ["opret faktura", "ny regning"]

    Returns:
        Formatted passages with source URLs, or a no-results message.
    """
    if _STUB:
        return _STUB_RESPONSE

    try:
        tasks = [asyncio.to_thread(_retrieve_raw, q) for q in queries]
        nested = await asyncio.gather(*tasks)
        all_results = [item for sublist in nested for item in sublist]
        unique = _deduplicate(all_results)
        if not unique:
            return "No relevant documentation found."
        return _format_passages(unique)
    except Exception as e:
        logger.error("KB retrieval error: %s", e)
        return "System error: unable to access documentation right now."
