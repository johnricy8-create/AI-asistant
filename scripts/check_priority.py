from __future__ import annotations

import json
from pathlib import Path

import httpx

CASES = [
    ("difference", "有差价怎么办", "20261025005"),
    ("product_broken", "商品坏了", "20261025002"),
    ("low_auto", "运费怎么算", "20261025004"),
    ("presale", "预售商品定金能退吗", "20261025001"),
    ("invoice", "怎么开发票", "20261025003"),
    ("large_product", "我买的扫地机器人用了两天就坏了，要退款", "20261025006"),
]


def main() -> None:
    output = {}
    with httpx.Client(timeout=30) as client:
        for name, message, order_id in CASES:
            response = client.post(
                "http://127.0.0.1:8000/api/v1/chat",
                json={
                    "session_id": f"check3-{name}",
                    "message": message,
                    "mode": "self_service",
                    "context": {"order_id": order_id},
                },
            )
            data = response.json()
            priority = data.get("priority") or {}
            output[name] = {
                "status": response.status_code,
                "channel": data.get("channel"),
                "need_human": data.get("need_human"),
                "auto_replied": data.get("auto_replied"),
                "priority": {
                    "level": priority.get("level"),
                    "category": priority.get("category"),
                    "category_label": priority.get("category_label"),
                    "handling": priority.get("handling"),
                    "handling_label": priority.get("handling_label"),
                    "reason": priority.get("reason"),
                    "source": priority.get("source"),
                },
                "reply": data.get("reply"),
                "skill_group": (data.get("handoff") or {}).get("skill_group"),
                "handoff_reason": (data.get("handoff") or {}).get("reason_code"),
            }
    Path("data/priority_check3.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("written")


if __name__ == "__main__":
    main()
