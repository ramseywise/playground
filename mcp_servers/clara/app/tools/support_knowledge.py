"""Support knowledge tool — queries a Bedrock Knowledge Base for sevdesk help docs."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import boto3

logger = logging.getLogger(__name__)

_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
_AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
_AWS_PROFILE = os.getenv("AWS_PROFILE")
_RERANK_MODEL_ARN = (
    f"arn:aws:bedrock:{os.getenv('AWS_REGION', 'eu-central-1')}::foundation-model/amazon.rerank-v1:0"
)
_SCORE_THRESHOLD = 0.4
_NUM_RESULTS = 15
_MAX_UNIQUE_RESULTS = 4

aws_session = boto3.Session(profile_name=_AWS_PROFILE, region_name=_AWS_REGION)
bedrock_agent_runtime = aws_session.client("bedrock-agent-runtime")


def _extract_url(location: dict[str, Any]) -> str | None:
    if "webLocation" in location:
        return location["webLocation"].get("url")
    if "s3Location" in location:
        return location["s3Location"].get("uri")
    if "confluenceLocation" in location:
        return location["confluenceLocation"].get("url")
    if "sharePointLocation" in location:
        return location["sharePointLocation"].get("url")
    if "salesforceLocation" in location:
        return location["salesforceLocation"].get("url")
    if "kendraDocumentLocation" in location:
        return location["kendraDocumentLocation"].get("uri")
    return None


def _extract_title_from_text(text: str) -> str | None:
    pipe_idx = text.find(" | ")
    if pipe_idx == -1 or pipe_idx > 120:
        return None
    candidate = text[:pipe_idx].strip()
    if not candidate or candidate[0].islower() or len(candidate) > 100:
        return None
    return candidate


def _build_passages(retrieval_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passages: list[dict[str, Any]] = []
    rank = 1
    for result in retrieval_results:
        score: float = result.get("score", 0.0)
        if score < _SCORE_THRESHOLD:
            continue
        text: str = result.get("content", {}).get("text", "").strip()
        if not text:
            continue
        location: dict = result.get("location", {})
        url: str | None = _extract_url(location)
        metadata: dict = result.get("metadata", {})
        title: str | None = metadata.get("title") or _extract_title_from_text(text) or url
        passages.append({
            "passage": rank,
            "score": round(score, 4),
            "url": url,
            "query": result.get("_query"),
            "title": title,
            "text": text,
        })
        rank += 1
    return passages


def _get_unique_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
    unique_results = []
    seen_fingerprints: set[str] = set()
    for res in sorted_results:
        location = res.get("location", {})
        url = _extract_url(location) or "no-url"
        content = res.get("content", {}).get("text", "")
        content_snippet = "".join(content[:200].split()).lower()
        fingerprint = f"{url}|{content_snippet}"
        if fingerprint not in seen_fingerprints:
            unique_results.append(res)
            seen_fingerprints.add(fingerprint)
    return unique_results


def _retrieve_from_kb_raw(query: str) -> list[dict]:
    if not _KB_ID:
        return []
    logger.info("Querying Bedrock KB with query: %s", query)
    response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=_KB_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": _NUM_RESULTS,
                "overrideSearchType": "HYBRID",
                "rerankingConfiguration": {
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration": {
                        "numberOfRerankedResults": 5,
                        "modelConfiguration": {"modelArn": _RERANK_MODEL_ARN},
                        "metadataConfiguration": {"selectionMode": "ALL"},
                    },
                },
            }
        },
    )
    results = response.get("retrievalResults", [])
    for r in results:
        r["_query"] = query
    return results


async def fetch_support_knowledge(queries: list[str]) -> list[dict[str, Any]]:
    """Search the official sevdesk support documentation.

    Args:
        queries: A list of 2-3 search terms or phrases (German or English).
                 Example: ["Rechnung erstellen", "neue Ausgangsrechnung"]

    Returns:
        List of passage dicts with passage rank, score, url, title, and text.
        Returns an empty list if the knowledge base is not configured or no
        results exceed the relevance threshold.
    """
    if not _KB_ID:
        return [{"error": "BEDROCK_KNOWLEDGE_BASE_ID is not configured"}]

    try:
        logger.debug("Starting parallel KB retrieval for: %s", queries)
        tasks = [asyncio.to_thread(_retrieve_from_kb_raw, q) for q in queries]
        results_nested = await asyncio.gather(*tasks)
        all_results = [item for sublist in results_nested for item in sublist]
        unique_results = _get_unique_results(all_results)
        if not unique_results:
            return []
        return _build_passages(unique_results[:_MAX_UNIQUE_RESULTS])
    except Exception as e:
        logger.error("Global KB retrieve error: %s", e, exc_info=True)
        return []
