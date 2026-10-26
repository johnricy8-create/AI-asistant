from __future__ import annotations

import re
from typing import Any

from app.data.seed import ORDERS

_ORDER_ID = re.compile(r"(?<!\d)(\d{8,})(?!\d)")


def extract_order_id(text: str | None, fallback: str | None = None) -> str | None:
    if text:
        match = _ORDER_ID.search(text)
        if match:
            return match.group(1)
    return fallback or None


def get_order(order_id: str | None) -> dict[str, Any] | None:
    if not order_id:
        return None
    return ORDERS.get(order_id)


def list_orders() -> list[dict[str, Any]]:
    return list(ORDERS.values())


def build_snapshot(order: dict[str, Any] | None) -> dict[str, Any] | None:
    if not order:
        return None
    return {
        "order_id": order["order_id"],
        "product_name": order["product_name"],
        "amount": order["amount"],
        "status": order["status"],
        "status_detail": order["status_detail"],
        "paid_at": order.get("paid_at"),
        "shipped_at": order.get("shipped_at"),
        "signed_at": order.get("signed_at"),
        "carrier": order.get("carrier"),
        "tracking_no": order.get("tracking_no"),
        "latest_event": order.get("latest_event"),
        "estimated_delivery": order.get("estimated_delivery"),
        "refund_status": order.get("refund_status"),
        "price_protection_until": order.get("price_protection_until"),
    }


def _order_status_text(order: dict[str, Any]) -> str:
    parts = [
        f"订单 {order['order_id']}（{order['product_name']}）当前状态为「{order['status']}」。",
        f"{order['status_detail']}。",
        f"最新进度：{order['latest_event']}。",
    ]
    if order["status"] == "已发货" and order.get("carrier"):
        parts.append(
            f"承运商：{order['carrier']}，运单号：{order['tracking_no']}，"
            f"预计 {order['estimated_delivery']} 送达。"
        )
    elif order["status"] == "待发货":
        parts.append(f"预计发货时间：{order['estimated_delivery']}。")
    elif order["status"] == "待支付":
        parts.append("订单尚未支付，支付后会尽快安排发货。")
    return "".join(parts)


def _logistics_text(order: dict[str, Any]) -> str:
    if not order.get("carrier"):
        return (
            f"订单 {order['order_id']} 目前还没有物流信息，当前状态为「{order['status']}」。"
            f"{order['status_detail']}。"
        )
    return (
        f"订单 {order['order_id']} 已由 {order['carrier']} 承运，运单号 {order['tracking_no']}。"
        f"最新物流节点：{order['latest_event']}，预计 {order['estimated_delivery']} 送达。"
    )


def _refund_text(order: dict[str, Any]) -> str:
    return (
        f"订单 {order['order_id']} 的退款状态为「{order['refund_status']}」。"
        "退款默认按原支付渠道退回：微信/支付宝一般 1–3 个工作日，银行卡一般 3–7 个工作日。"
    )


def query_tool(intent: str, order_id: str | None) -> dict[str, Any]:
    """Read-only order/logistics/refund lookup used by the Tool channel."""
    order = get_order(order_id)
    if not order:
        return {
            "found": False,
            "intent": intent,
            "order_id": order_id,
            "snapshot": None,
            "answer": (
                "我没有查到这笔订单，可能是订单号有误或超出了当前查询范围。"
                "请您核对订单号，或我帮您转人工进一步查询。"
            ),
            "policy_query": "订单查询 物流 退款",
            "detail": f"未找到订单 {order_id or '（空）'}",
        }

    if intent == "logistics":
        answer = _logistics_text(order)
        policy_query = "物流异常 签收 配送 丢件"
    elif intent == "refund_status":
        answer = _refund_text(order)
        policy_query = "退款到账 原路退回 到账时间"
    else:
        answer = _order_status_text(order)
        policy_query = "发货时效 预售 大促 48小时"

    return {
        "found": True,
        "intent": intent,
        "order_id": order["order_id"],
        "snapshot": build_snapshot(order),
        "answer": answer,
        "policy_query": policy_query,
        "detail": f"订单 {order['order_id']} · 状态：{order['status']}",
    }
