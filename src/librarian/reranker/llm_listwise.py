from __future__ import annotations

import json

from core.clients.llm import LLMClient
from librarian.schemas.chunks import GradedChunk, RankedChunk
from core.logging import get_logger

log = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a relevance ranking assistant. "
    "Given a query and a numbered list of documents, rank them by relevance to the query. "
    "Return ONLY a JSON array (no prose) in this exact format:\n"
    '[{"rank": 1, "doc_index": <int>, "relevance_score": <float 0-1>}, ...]'
)


class LLMListwiseReranker:
    """Reranker that uses an LLM (Haiku) for listwise relevance ranking.

    Fallback behaviour:
    - Total parse failure → return input order with relevance_score=0.5
    - Partial parse (fewer indices than chunks) → append missing chunks at
      score 0.5 in original input order so confidence_score is never computed
      from an incomplete list.

    Used for experiments; cross_encoder is the prod default.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def rerank(
        self,
        query: str,
        chunks: list[GradedChunk],
        top_k: int = 3,
    ) -> list[RankedChunk]:
        if not chunks:
            return []

        doc_list = "\n".join(
            f"[{i}] {gc.chunk.text[:300]}" for i, gc in enumerate(chunks)
        )
        user_msg = f"Query: {query}\n\nDocuments:\n{doc_list}"

        try:
            raw = await self._llm.generate(
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                max_tokens=1024,
            )
            ranked = self._parse(raw, chunks)
        except Exception as exc:
            log.error("reranker.llm_listwise.failed", error=str(exc))
            ranked = self._fallback(chunks)

        return ranked[:top_k]

    def _parse(self, raw: str, chunks: list[GradedChunk]) -> list[RankedChunk]:
        try:
            # Strip markdown code fences if present
            cleaned = (
                raw.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            items: list[dict] = json.loads(cleaned)
            seen_indices: set[int] = set()
            result: list[RankedChunk] = []

            for item in items:
                idx = int(item["doc_index"])
                score = float(item["relevance_score"])
                if 0 <= idx < len(chunks) and idx not in seen_indices:
                    seen_indices.add(idx)
                    result.append(
                        RankedChunk(
                            chunk=chunks[idx].chunk,
                            relevance_score=min(max(score, 0.0), 1.0),
                            rank=len(result) + 1,
                        )
                    )

            # Partial parse fallback: append missing indices at 0.5
            missing_start_rank = len(result) + 1
            for i, gc in enumerate(chunks):
                if i not in seen_indices:
                    result.append(
                        RankedChunk(
                            chunk=gc.chunk, relevance_score=0.5, rank=missing_start_rank
                        )
                    )
                    missing_start_rank += 1

            if not result:
                return self._fallback(chunks)
            return result

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.error("reranker.llm_listwise.parse_error", error=str(exc))
            return self._fallback(chunks)

    def _fallback(self, chunks: list[GradedChunk]) -> list[RankedChunk]:
        return [
            RankedChunk(chunk=gc.chunk, relevance_score=0.5, rank=i + 1)
            for i, gc in enumerate(chunks)
        ]
