from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class SiliconFlowEmbedder:
    name = "SiliconFlow"

    def __init__(self, settings: Settings):
        if not settings.siliconflow_api_key.strip():
            raise EmbeddingError("SILICONFLOW_API_KEY is empty")
        self.api_key = settings.siliconflow_api_key.strip()
        self.base_url = settings.siliconflow_base_url.rstrip("/")
        self.model = settings.siliconflow_embedding_model
        self.timeout = settings.embedding_timeout_seconds
        self.name = f"SiliconFlow/{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        batch_size = 32
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                response = client.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json={
                        "model": self.model,
                        "input": batch,
                        "encoding_format": "float",
                    },
                )
                if response.status_code != 200:
                    raise EmbeddingError(
                        f"SiliconFlow embedding failed: HTTP {response.status_code} "
                        f"{response.text[:300]}"
                    )
                data = response.json()
                items = sorted(
                    data.get("data", []),
                    key=lambda item: item.get("index", 0),
                )
                batch_vectors = [item.get("embedding", []) for item in items]
                if len(batch_vectors) != len(batch):
                    raise EmbeddingError(
                        "SiliconFlow embedding count mismatch: "
                        f"expected {len(batch)}, got {len(batch_vectors)}"
                    )
                vectors.extend(batch_vectors)
        return vectors


class LocalSentenceTransformerEmbedder:
    name = "SentenceTransformers"

    def __init__(self, settings: Settings):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = settings.local_embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.name = f"SentenceTransformers/{self.model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]


def build_embedder(settings: Settings) -> Any | None:
    provider = settings.embedding_provider.strip().lower()
    if provider == "siliconflow":
        if not settings.siliconflow_enabled:
            logger.warning("embedding_provider=siliconflow but API key is missing")
            return None
        try:
            return SiliconFlowEmbedder(settings)
        except Exception:  # noqa: BLE001
            logger.exception("failed to initialize SiliconFlow embedder")
            return None

    if provider == "local":
        try:
            return LocalSentenceTransformerEmbedder(settings)
        except Exception:  # noqa: BLE001
            logger.exception("failed to initialize local SentenceTransformer embedder")
            return None

    # auto: prefer SiliconFlow, then local
    if settings.siliconflow_enabled:
        try:
            return SiliconFlowEmbedder(settings)
        except Exception:  # noqa: BLE001
            logger.exception("failed to initialize SiliconFlow embedder")
    try:
        return LocalSentenceTransformerEmbedder(settings)
    except Exception:  # noqa: BLE001
        logger.exception("failed to initialize local SentenceTransformer embedder")
        return None
