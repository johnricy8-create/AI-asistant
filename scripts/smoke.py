from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.data.seed import SCENARIOS
from app.main import create_app
from fastapi.testclient import TestClient


def main() -> None:
    settings = Settings(
        database_url="sqlite:///./data/smoke.db",
        llm_mode="mock",
    )
    app = create_app(settings)
    output = {}
    with TestClient(app) as client:
        for scenario in SCENARIOS:
            session_id = f"smoke-{scenario['id']}"
            if scenario.get("seed_messages"):
                client.post(
                    "/api/v1/demo/seed",
                    json={
                        "session_id": session_id,
                        "messages": scenario["seed_messages"],
                    },
                )
            response = client.post(
                "/api/v1/chat",
                json={
                    "session_id": session_id,
                    "message": scenario["message"],
                    "context": {"order_id": scenario["order_id"]},
                },
            )
            data = response.json()
            output[scenario["id"]] = data
            print(
                scenario["id"],
                response.status_code,
                data.get("channel"),
                data.get("confidence"),
                data.get("latency_ms"),
            )
    Path("data/smoke_output.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
