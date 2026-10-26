from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
EMBEDDING_MODEL = os.environ.get("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get(
    "SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"
)


def main() -> None:
    if not API_KEY:
        raise SystemExit("SILICONFLOW_API_KEY is empty")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    output: dict = {
        "base_url": BASE_URL,
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
    }

    with httpx.Client(timeout=60) as client:
        embedding_response = client.post(
            f"{BASE_URL.rstrip('/')}/embeddings",
            headers=headers,
            json={
                "model": EMBEDDING_MODEL,
                "input": ["退货政策是什么", "物流时效怎么算"],
                "encoding_format": "float",
            },
        )
        output["embedding_status"] = embedding_response.status_code
        if embedding_response.status_code == 200:
            data = embedding_response.json()
            vectors = data.get("data") or []
            output["embedding_count"] = len(vectors)
            output["embedding_dim"] = len(vectors[0].get("embedding", [])) if vectors else 0
            output["embedding_usage"] = data.get("usage")
        else:
            output["embedding_error"] = embedding_response.text[:1000]

        rerank_response = client.post(
            f"{BASE_URL.rstrip('/')}/rerank",
            headers=headers,
            json={
                "model": RERANK_MODEL,
                "query": "怎么退货",
                "documents": [
                    "商品签收后 7 天内，商品完好、不影响二次销售的，可申请无理由退货。",
                    "今天天气不错，适合出门散步。",
                ],
                "top_n": 2,
                "return_documents": False,
            },
        )
        output["rerank_status"] = rerank_response.status_code
        if rerank_response.status_code == 200:
            output["rerank"] = rerank_response.json()
        else:
            output["rerank_error"] = rerank_response.text[:1000]

    Path("data/siliconflow_test.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("written")


if __name__ == "__main__":
    main()
