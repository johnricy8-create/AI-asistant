from __future__ import annotations

import json
import time
from pathlib import Path

from app.config import Settings
from app.rag.factory import build_chunks, build_reranker, build_retriever
from app.rag.index import HybridRetriever


def brief(hits: list[dict]) -> list[dict]:
    return [
        {
            "title": hit["title"],
            "section": hit["section"],
            "score": hit["score"],
        }
        for hit in hits
    ]


def main() -> None:
    settings = Settings()
    chunks = build_chunks()
    lexical = HybridRetriever(chunks)
    vector = build_retriever(settings)
    reranker = build_reranker(settings)

    queries = [
        "预售商品定金能退吗",
        "怎么退货",
        "商品坏了怎么办",
        "退款到账了吗",
        "运费怎么算",
        "耳机有质量问题能退吗",
        "大额商品故障怎么处理",
    ]

    output = {
        "retriever": getattr(vector, "name", ""),
        "reranker": getattr(reranker, "name", ""),
        "recall_top_k": settings.recall_top_k,
        "rerank_top_k": settings.rerank_top_k,
        "queries": {},
    }

    for query in queries:
        start = time.perf_counter()
        lexical_hits = lexical.search(query, top_k=5)
        lexical_ms = int((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        recall_hits = vector.search(query, top_k=settings.recall_top_k)
        recall_ms = int((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        reranked_hits = reranker.rerank(
            query, recall_hits, top_k=settings.rerank_top_k
        )
        rerank_ms = int((time.perf_counter() - start) * 1000)

        output["queries"][query] = {
            "lexical_ms": lexical_ms,
            "chroma_recall_ms": recall_ms,
            "rerank_ms": rerank_ms,
            "recall_count": len(recall_hits),
            "lexical_top5": brief(lexical_hits),
            "vector_recall_top5": brief(recall_hits[:5]),
            "reranked_top5": brief(reranked_hits),
        }

    Path("data/effect_compare.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("written")


if __name__ == "__main__":
    main()
