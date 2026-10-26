from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.rag.factory import build_reranker, build_retriever


def main() -> None:
    settings = Settings()
    retriever = build_retriever(settings)
    reranker = build_reranker(settings)
    output: dict = {
        "retrieval_backend": getattr(retriever, "name", ""),
        "embedding_model": getattr(retriever, "embedding_model", ""),
        "reranker": getattr(reranker, "name", ""),
        "recall_top_k": settings.recall_top_k,
        "rerank_top_k": settings.rerank_top_k,
        "queries": {},
    }

    for query in ["预售商品定金能退吗", "怎么退货", "商品坏了怎么办"]:
        hits = retriever.search(query, top_k=settings.recall_top_k)
        reranked = reranker.rerank(query, hits, top_k=settings.rerank_top_k)
        output["queries"][query] = {
            "recall_count": len(hits),
            "rerank_count": len(reranked),
            "hits": [
                {
                    "title": hit["title"],
                    "section": hit["section"],
                    "score": hit["score"],
                    "retrieval_score": hit.get("retrieval_score"),
                }
                for hit in reranked
            ],
        }

    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "session_id": "chroma-live",
                "message": "预售商品定金能退吗",
                "context": {"order_id": "20261025001"},
            },
        )
        data = response.json()
        output["chat"] = {
            "status": response.status_code,
            "channel": data.get("channel"),
            "priority": data.get("priority"),
            "citations": [item.get("title") for item in data.get("citations", [])],
            "trace": [
                {
                    "name": step.get("name"),
                    "label": step.get("label"),
                    "detail": step.get("detail"),
                    "latency_ms": step.get("latency_ms"),
                }
                for step in data.get("trace", [])
            ],
        }

    Path("data/chroma_live_test.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("written")


if __name__ == "__main__":
    main()
