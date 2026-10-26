from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.data.seed import KB_SEED
from app.rag.chunking import build_chunks_from_seed
from app.rag.chroma_store import ChromaRetriever, FallbackRetriever
from app.rag.embeddings import build_embedder
from app.rag.index import HybridRetriever
from app.rag.rerank import build_reranker

logger = logging.getLogger(__name__)


def build_chunks() -> list[dict[str, Any]]:
    return build_chunks_from_seed(KB_SEED)


def build_retriever(settings: Settings) -> Any:
    chunks = build_chunks()
    lexical = HybridRetriever(chunks)
    lexical.name = "BM25 + 字符 n-gram"

    backend = settings.retrieval_backend.strip().lower()
    if backend == "lexical":
        return lexical

    if backend in {"chroma", "auto"}:
        embedder = build_embedder(settings)
        if embedder is None:
            logger.warning("embedder unavailable; using lexical retrieval")
            return lexical
        try:
            chroma = ChromaRetriever(settings, chunks, embedder)
            return FallbackRetriever(chroma, lexical)
        except Exception:  # noqa: BLE001 - the demo must still start
            logger.exception("ChromaDB initialization failed; using lexical retrieval")
            return lexical

    logger.warning("unknown retrieval_backend=%s; using lexical", backend)
    return lexical


__all__ = ["build_chunks", "build_retriever", "build_reranker"]
