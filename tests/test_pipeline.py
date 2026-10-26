from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.orchestration.priority import PriorityEngine


@pytest.fixture()
def client(tmp_path: Path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        llm_mode="mock",
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def chat(
    client: TestClient,
    message: str,
    *,
    session_id: str = "test-session",
    order_id: str | None = None,
    mode: str = "copilot",
) -> dict:
    response = client.post(
        "/api/v1/chat",
        json={
            "session_id": session_id,
            "message": message,
            "mode": mode,
            "context": {"order_id": order_id},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_health_and_home(client: TestClient) -> None:
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    data = health.json()
    assert data["status"] == "ok"
    assert data["model_mode"] == "mock"
    assert data["faq_count"] >= 6
    assert data["kb_count"] >= 30

    home = client.get("/")
    assert home.status_code == 200
    assert "智能客服 Copilot" in home.text


def test_faq_channel(client: TestClient) -> None:
    data = chat(client, "怎么退货", order_id="20261025003")
    assert data["channel"] == "faq"
    assert data["confidence"] >= 0.9
    assert data["citations"]
    assert data["debug"]["faq"]["passed"] is True
    assert data["acknowledgement"] == "您好，已收到您的问题，我先帮您查询一下，请稍候。"
    assert any(step["name"] == "holding" for step in data["trace"])


def test_rag_channel(client: TestClient) -> None:
    data = chat(client, "预售商品定金能退吗", order_id="20261025001")
    assert data["channel"] == "rag"
    assert data["confidence"] > 0.5
    assert any("预售" in citation["title"] for citation in data["citations"])
    assert data["debug"]["retrieval"]


def test_tool_channel(client: TestClient) -> None:
    data = chat(client, "我的订单还没发货怎么办", order_id="20261025001")
    assert data["channel"] == "tool"
    assert data["order_snapshot"]["status"] == "待发货"
    assert data["debug"]["tool"]["found"] is True
    assert data["need_human"] is False


def test_refund_status_tool_channel(client: TestClient) -> None:
    data = chat(client, "退款到账了吗", order_id="20261025003")
    assert data["channel"] == "tool"
    assert "退款" in data["reply"]
    assert data["debug"]["tool"]["intent"] == "refund_status"


def test_human_handoff_for_price_protection(client: TestClient) -> None:
    data = chat(client, "我要退差价", order_id="20261025005")
    assert data["channel"] == "human"
    assert data["need_human"] is True
    assert data["handoff"]["reason_code"] == "price_protection"
    assert any("价保" in citation["title"] for citation in data["citations"])


def test_multi_turn_rewrite(client: TestClient) -> None:
    session_id = "test-follow-up"
    seed = client.post(
        "/api/v1/demo/seed",
        json={
            "session_id": session_id,
            "messages": [
                {
                    "role": "customer",
                    "content": "我买的是预售商品，什么时候发货？",
                }
            ],
        },
    )
    assert seed.status_code == 200

    data = chat(
        client,
        "那定金能退吗",
        session_id=session_id,
        order_id="20261025001",
    )
    assert data["channel"] == "rag"
    assert "预售" in data["debug"]["rewritten_query"]
    assert data["debug"]["context_dependent"] is True


def test_smalltalk(client: TestClient) -> None:
    data = chat(client, "你好")
    assert data["channel"] == "faq"
    assert "智能客服" in data["reply"]


def test_feedback_and_handoff_endpoints(client: TestClient) -> None:
    data = chat(client, "怎么开发票", order_id="20261025003")
    feedback = client.post(
        "/api/v1/feedback",
        json={
            "message_id": data["message_id"],
            "type": "helpful",
        },
    )
    assert feedback.status_code == 200
    assert feedback.json()["ok"] is True

    handoff = client.post(
        "/api/v1/handoff",
        json={
            "conversation_id": "test-session",
            "reason_code": "price_protection",
            "priority": "high",
            "skill_group": "after_sales",
            "context": {"summary": "测试转人工"},
        },
    )
    assert handoff.status_code == 200
    payload = handoff.json()
    assert payload["handoff_id"].startswith("ho-")
    assert payload["status"] == "queued"


def test_out_of_scope_falls_back_to_human(client: TestClient) -> None:
    data = chat(client, "今天天气怎么样")
    assert data["channel"] == "human"
    assert data["need_human"] is True
    assert any(flag["code"] == "out_of_scope" for flag in data["risk_flags"])
    assert data["handoff"]["reason_code"] == "out_of_scope"


def test_low_priority_auto_reply(client: TestClient) -> None:
    data = chat(
        client,
        "运费怎么算",
        order_id="20261025004",
        mode="self_service",
    )
    assert data["priority"]["level"] == "P3"
    assert data["priority"]["handling"] == "auto_reply"
    assert data["auto_replied"] is True
    assert data["channel"] == "faq"
    assert data["need_human"] is False


def test_low_priority_copilot_does_not_auto_send(client: TestClient) -> None:
    data = chat(client, "运费怎么算", mode="copilot")
    assert data["priority"]["level"] == "P3"
    assert data["priority"]["handling"] == "auto_reply"
    assert data["auto_replied"] is False


def test_product_issue_large_amount_fallback(client: TestClient) -> None:
    data = chat(
        client,
        "我买的扫地机器人用了两天就坏了，要退款",
        order_id="20261025006",
        mode="self_service",
    )
    assert data["priority"]["level"] in {"P1", "P0"}
    assert data["priority"]["handling"] in {"fallback_human", "human"}
    assert data["priority"]["category"] == "product_quality"
    assert data["priority"]["product_issue"] is True
    assert data["priority"]["large_amount"] is True
    assert data["auto_replied"] is False
    assert data["need_human"] is True
    assert data["channel"] == "human"
    assert "人工" in data["reply"] or "转接" in data["reply"]
    assert "凭证" in data["reply"]
    assert "平台" in data["reply"]
    assert data["handoff"]["reason_code"] in {"product_issue", "large_amount"}
    assert "兜底" in data["priority"]["reason"] or "商品" in data["priority"]["reason"]


def test_large_amount_money_action_fallback(client: TestClient) -> None:
    data = chat(
        client,
        "这笔订单我要退款",
        order_id="20261025006",
        mode="self_service",
    )
    assert data["priority"]["level"] in {"P1", "P0"}
    assert data["priority"]["large_amount"] is True
    assert data["priority"]["handling"] in {"fallback_human", "human"}
    assert data["auto_replied"] is False
    assert data["need_human"] is True


def test_presale_high_priority_copilot_not_auto(client: TestClient) -> None:
    data = chat(
        client,
        "预售商品定金能退吗",
        order_id="20261025001",
        mode="self_service",
    )
    assert data["priority"]["level"] == "P1"
    assert data["priority"]["handling"] == "copilot"
    assert data["auto_replied"] is False
    assert data["channel"] == "rag"
    assert "平台" in data["reply"] or "审核" in data["reply"]


def test_price_protection_priority_fallback(client: TestClient) -> None:
    data = chat(
        client,
        "我要退差价",
        order_id="20261025005",
        mode="self_service",
    )
    assert data["priority"]["level"] == "P1"
    assert data["priority"]["handling"] == "fallback_human"
    assert data["priority"]["category"] == "money"
    assert data["auto_replied"] is False
    assert data["need_human"] is True
    assert "平台" in data["reply"]
    assert "核实" in data["reply"]


def test_difference_rule_question_also_falls_back(client: TestClient) -> None:
    data = chat(
        client,
        "差价怎么算",
        order_id="20261025005",
        mode="self_service",
    )
    assert data["priority"]["level"] == "P1"
    assert data["priority"]["category"] == "money"
    assert data["priority"]["handling"] == "fallback_human"
    assert data["auto_replied"] is False


def test_difference_keyword_triggers_fallback(client: TestClient) -> None:
    data = chat(
        client,
        "有差价怎么办",
        order_id="20261025005",
        mode="self_service",
    )
    assert data["priority"]["level"] == "P1"
    assert data["priority"]["category"] == "money"
    assert data["priority"]["handling"] == "fallback_human"
    assert data["auto_replied"] is False
    assert "差价" in data["priority"]["reason"]


def test_product_broken_keyword_triggers_fallback(client: TestClient) -> None:
    data = chat(
        client,
        "商品坏了",
        order_id="20261025002",
        mode="self_service",
    )
    assert data["priority"]["level"] == "P1"
    assert data["priority"]["category"] == "product_quality"
    assert data["priority"]["handling"] == "fallback_human"
    assert data["auto_replied"] is False
    assert "商品" in data["priority"]["reason"]


def test_priority_engine_can_be_raised_by_llm() -> None:
    async def classifier(**_: object) -> dict:
        return {
            "priority": "P1",
            "product_issue": True,
            "reason": "大模型识别到商品质量问题",
            "confidence": 0.88,
        }

    engine = PriorityEngine(large_amount_threshold=500)
    result = asyncio.run(
        engine.assess(
            question="运费怎么算",
            order_snapshot=None,
            risk_flags=[],
            intent="general",
            personal_order=False,
            classifier=classifier,
        )
    )
    assert result.level == "P1"
    assert result.handling == "fallback_human"
    assert result.product_issue is True
    assert result.category == "product_quality"
    assert result.source == "rule+llm"


def test_llm_can_require_human_for_high_priority_policy() -> None:
    async def classifier(**_: object) -> dict:
        return {
            "priority": "P1",
            "category": "资金安全",
            "need_human": True,
            "auto_reply": False,
            "reason": "定金争议需要人工核实",
            "confidence": 0.86,
        }

    engine = PriorityEngine(large_amount_threshold=500)
    result = asyncio.run(
        engine.assess(
            question="预售商品定金能退吗",
            order_snapshot={"amount": 399.0},
            risk_flags=[],
            intent="presale_deposit",
            personal_order=False,
            classifier=classifier,
        )
    )
    assert result.level == "P1"
    assert result.handling == "fallback_human"
    assert result.category == "money"
    assert result.source == "rule+llm"


def test_llm_can_deny_p3_auto_reply() -> None:
    async def classifier(**_: object) -> dict:
        return {
            "priority": "P3",
            "category": "一般咨询",
            "need_human": False,
            "auto_reply": False,
            "reason": "建议人工确认",
            "confidence": 0.8,
        }

    engine = PriorityEngine(large_amount_threshold=500)
    result = asyncio.run(
        engine.assess(
            question="运费怎么算",
            order_snapshot=None,
            risk_flags=[],
            intent="general",
            personal_order=False,
            classifier=classifier,
        )
    )
    assert result.level == "P3"
    assert result.handling == "copilot"
    assert result.source == "rule+llm"


def test_llm_can_allow_p3_auto_reply() -> None:
    async def classifier(**_: object) -> dict:
        return {
            "priority": "P3",
            "category": "一般咨询",
            "need_human": False,
            "auto_reply": True,
            "reason": "低风险规则咨询",
            "confidence": 0.9,
        }

    engine = PriorityEngine(large_amount_threshold=500)
    result = asyncio.run(
        engine.assess(
            question="运费怎么算",
            order_snapshot=None,
            risk_flags=[],
            intent="general",
            personal_order=False,
            classifier=classifier,
        )
    )
    assert result.level == "P3"
    assert result.handling == "auto_reply"
    assert result.source == "rule+llm"


def test_priority_rule_floor_cannot_be_lowered_by_llm() -> None:
    called = {"value": False}

    async def classifier(**_: object) -> dict:
        called["value"] = True
        return {"priority": "P3", "reason": "试图降低", "confidence": 0.99}

    engine = PriorityEngine(large_amount_threshold=500)
    result = asyncio.run(
        engine.assess(
            question="我要退差价",
            order_snapshot={"amount": 359.0},
            risk_flags=[],
            intent="price_protection",
            personal_order=False,
            classifier=classifier,
        )
    )
    assert result.level == "P1"
    assert result.handling == "fallback_human"
    assert result.source == "rule"
    assert called["value"] is False
