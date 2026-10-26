from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def main() -> None:
    settings = Settings()
    app = create_app(settings)
    cases = [
        {
            "id": "live-rag",
            "message": "预售商品定金能退吗",
            "order_id": "20261025001",
        },
        {
            "id": "live-tool",
            "message": "我的订单还没发货怎么办",
            "order_id": "20261025001",
        },
    ]
    output = {}
    with TestClient(app) as client:
        for case in cases:
            response = client.post(
                "/api/v1/chat",
                json={
                    "session_id": case["id"],
                    "message": case["message"],
                    "context": {"order_id": case["order_id"]},
                },
            )
            data = response.json()
            output[case["id"]] = data
            print(
                case["id"],
                response.status_code,
                data.get("channel"),
                data.get("model"),
                data.get("model_mode"),
                data.get("latency_ms"),
            )
            time.sleep(3)
    Path("data/live_chat.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
