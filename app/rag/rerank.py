from __future__ import annotations

import logging
import math
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class RerankError(RuntimeError):
    pass


class SiliconFlowReranker:
    active = True
    name = "SiliconFlow"

    def __init__(self, settings: Settings):
        if not settings.siliconflow_api_key.strip():
            raise RerankError("SILICONFLOW_API_KEY is empty")
        self.api_key = settings.siliconflow_api_key.strip()
        self.base_url = settings.siliconflow_base_url.rstrip("/")
        self.model = settings.siliconflow_rerank_model
        self.timeout = settings.rerank_timeout_seconds
        self.name = f"SiliconFlow/{self.model}"

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not hits:
            return []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": [hit["content"] for hit in hits],
            "top_n": min(top_k, len(hits)),
            "return_documents": False,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/rerank",
                headers=headers,
                json=payload,
            )
        if response.status_code != 200:
            raise RerankError(
                f"SiliconFlow rerank failed: HTTP {response.status_code} "
                f"{response.text[:300]}"
            )
        data = response.json()
        results = data.get("results") or []
        reranked: list[dict[str, Any]] = []
        for result in results:
            index = result.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(hits):
                continue
            hit = dict(hits[index])
            score = float(result.get("relevance_score", 0.0))
            hit["retrieval_score"] = hit.get("score", 0.0)
            hit["rerank_score"] = score
            hit["score"] = round(max(0.0, min(1.0, score)), 4)
            reranked.append(hit)
        reranked.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
        return reranked[:top_k]


class LocalCrossEncoderReranker:
    active = True
    name = "CrossEncoder"

    def __init__(self, settings: Settings):
        from sentence_transformers import CrossEncoder  # lazy import

        self.model_name = settings.local_reranker_model
        self.model = CrossEncoder(self.model_name)
        self.name = f"CrossEncoder/{self.model_name}"

    @staticmethod
    def _sigmoid(value: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-value))
        except OverflowError:
            return 0.0 if value < 0 else 1.0

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not hits:
            return []
        pairs = [(query, hit["content"]) for hit in hits]
        scores = self.model.predict(pairs, batch_size=8)
        reranked: list[dict[str, Any]] = []
        for hit, raw_score in zip(hits, scores):
            item = dict(hit)
            score = self._sigmoid(float(raw_score))
            item["retrieval_score"] = item.get("score", 0.0)
            item["rerank_score"] = score
            item["score"] = round(score, 4)
            reranked.append(item)
        reranked.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
        return reranked[:top_k]


class NoopReranker:
    active = False
    name = "None"

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return sorted(hits, key=lambda item: item.get("score", 0.0), reverse=True)[
            :top_k
        ]


def build_reranker(settings: Settings) -> Any:
    provider = settings.rerank_provider.strip().lower()
    if provider == "none":
        return NoopReranker()

    if provider == "siliconflow":
        if not settings.siliconflow_enabled:
            logger.warning("rerank_provider=siliconflow but API key is missing")
            return NoopReranker()
        try:
            return SiliconFlowReranker(settings)
        except Exception:  # noqa: BLE001
            logger.exception("failed to initialize SiliconFlow reranker")
            return NoopReranker()

    if provider == "local":
        try:
            return LocalCrossEncoderReranker(settings)
        except Exception:  # noqa: BLE001
            logger.exception("failed to initialize local CrossEncoder reranker")
            return NoopReranker()

    # auto: prefer SiliconFlow, then local
    if settings.siliconflow_enabled:
        try:
            return SiliconFlowReranker(settings)
        except Exception:  # noqa: BLE001
            logger.exception("failed to initialize SiliconFlow reranker")
    try:
        return LocalCrossEncoderReranker(settings)
    except Exception:  # noqa: BLE001
        logger.exception("failed to initialize local CrossEncoder reranker")
        return NoopReranker()
