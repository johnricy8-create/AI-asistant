from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.text import contains_any, normalize_text
from app.data.orders import extract_order_id

HUMAN_WORDS = ["转人工", "人工客服", "找人工", "真人", "人工服务", "我要人工"]
COMPLAINT_WORDS = ["投诉", "起诉", "曝光", "12315", "律师", "骗子", "垃圾", "气死", "差评", "媒体"]
NEGATIVE_WORDS = ["气死", "垃圾", "骗子", "恶心", "失望", "差评", "无语", "烦死"]
MONEY_WORDS = ["退款", "退钱", "退差价", "退定金", "赔偿", "补偿", "价保"]
INQUIRY_WORDS = [
    "怎么",
    "如何",
    "多久",
    "什么时候",
    "规则",
    "政策",
    "流程",
    "多少",
    "吗",
    "?",
    "是否",
    "能不能",
    "可不可以",
    "到哪",
    "进度",
]
ADDRESS_PATTERNS = [
    r"(我要|帮我|给我|麻烦|请).{0,6}(改|修改|换).{0,6}(地址|收货地址)",
    r"改地址",
    r"修改地址",
    r"改收货地址",
    r"修改收货地址",
]
CANCEL_PATTERNS = [
    r"(我要|帮我|给我|麻烦|请).{0,6}取消.{0,4}订单",
    r"取消订单",
]
PERSONAL_ORDER_WORDS = [
    "我的订单",
    "这笔订单",
    "这个订单",
    "我的包裹",
    "我的快递",
    "我的退款",
    "退款到账",
    "退款进度",
    "到哪",
    "签收",
]
DOMAIN_WORDS = [
    "订单",
    "发货",
    "物流",
    "快递",
    "包裹",
    "退货",
    "退款",
    "运费",
    "邮费",
    "包邮",
    "发票",
    "开票",
    "预售",
    "定金",
    "尾款",
    "价保",
    "差价",
    "地址",
    "签收",
    "破损",
    "丢件",
    "会员",
    "优惠券",
    "支付",
    "商品",
    "售后",
    "客服",
    "大促",
    "配送",
    "到货",
    "改址",
    "退换",
    "退货",
    "保修",
    "积分",
]
CONTEXT_MARKERS = [
    "这个",
    "那个",
    "它",
    "那",
    "这",
    "也是",
    "还",
    "呢",
    "能吗",
    "可以吗",
    "真的吗",
    "然后",
    "刚才",
]


def _flag(code: str, label: str, level: str = "warn") -> dict[str, str]:
    return {"code": code, "label": label, "level": level}


@dataclass
class RiskAssessment:
    flags: list[dict[str, str]] = field(default_factory=list)
    is_high_risk: bool = False
    reason_code: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "flags": self.flags,
            "is_high_risk": self.is_high_risk,
            "reason_code": self.reason_code,
        }


@dataclass
class RouteDecision:
    channel: str  # human | tool | knowledge
    intent: str
    reason: str
    risk: RiskAssessment
    order_id: str | None
    context_dependent: bool
    rewritten_query: str
    personal_order: bool
    tool_intent: str | None = None


def is_inquiry(text: str) -> bool:
    return contains_any(text, INQUIRY_WORDS)


def assess_risk(message: str) -> RiskAssessment:
    text = normalize_text(message)
    flags: list[dict[str, str]] = []
    high = False
    reason: str | None = None

    if contains_any(text, HUMAN_WORDS):
        flags.append(_flag("human_request", "用户要求人工", "warn"))
        high = True
        reason = reason or "user_request"

    if contains_any(text, COMPLAINT_WORDS):
        flags.append(_flag("complaint", "投诉/舆情风险", "danger"))
        high = True
        reason = reason or "complaint"

    inquiry = is_inquiry(text)
    # "退款到账了吗 / 退款进度" is a read-only status query, not a write action.
    status_query = contains_any(text, ["到账", "进度", "到哪"])

    has_money = contains_any(text, MONEY_WORDS)
    if has_money and not inquiry and not status_query:
        flags.append(_flag("money", "涉及退款/差价/赔偿", "danger"))
        high = True
        if reason in {None, "user_request"}:
            reason = "refund_request"
        if contains_any(text, ["退差价", "价保"]):
            reason = "price_protection"

    if any(re.search(pattern, text) for pattern in ADDRESS_PATTERNS) and not inquiry:
        flags.append(_flag("address_change", "修改收货地址", "warn"))
        high = True
        reason = reason or "address_change"

    if any(re.search(pattern, text) for pattern in CANCEL_PATTERNS):
        flags.append(_flag("order_cancel", "取消订单", "warn"))
        high = True
        reason = reason or "order_cancel"

    if contains_any(text, NEGATIVE_WORDS) and not contains_any(text, COMPLAINT_WORDS):
        flags.append(_flag("negative_sentiment", "负面情绪", "warn"))

    # De-duplicate flags by code.
    seen: set[str] = set()
    unique_flags: list[dict[str, str]] = []
    for item in flags:
        if item["code"] in seen:
            continue
        seen.add(item["code"])
        unique_flags.append(item)

    return RiskAssessment(flags=unique_flags, is_high_risk=high, reason_code=reason)


def detect_intent(message: str) -> str:
    text = normalize_text(message)
    if contains_any(text, ["预售", "定金"]):
        return "presale_deposit"
    if contains_any(text, ["退差价", "价保"]):
        return "price_protection"
    if contains_any(text, ["发票", "开票"]):
        return "invoice"
    if contains_any(text, ["地址", "收货地址", "改址"]):
        return "address_change"
    if contains_any(text, ["退款", "退钱"]) and contains_any(text, ["到账", "进度", "到哪"]):
        return "refund_status"
    if contains_any(text, ["退货", "七天无理由", "能退吗", "可以退吗", "能不能退"]):
        return "return_policy"
    if contains_any(text, ["物流", "快递", "运单", "到哪", "签收", "派送"]):
        return "logistics"
    if contains_any(text, ["发货", "催单", "催发货"]):
        return "shipping_time"
    if contains_any(text, ["订单"]):
        return "order_status"
    return "general"


def is_context_dependent(message: str) -> bool:
    text = normalize_text(message)
    if len(text) > 16:
        return False
    return contains_any(text, CONTEXT_MARKERS)


def is_personal_order_query(message: str) -> bool:
    text = normalize_text(message)
    if extract_order_id(text):
        return True
    return contains_any(text, PERSONAL_ORDER_WORDS)


def has_domain_signal(message: str) -> bool:
    return contains_any(normalize_text(message), DOMAIN_WORDS)


def _tool_intent(intent: str) -> str:
    if intent == "logistics":
        return "logistics"
    if intent == "refund_status":
        return "refund_status"
    return "order_status"


class IntentRouter:
    def route(
        self,
        *,
        message: str,
        previous_user_message: str | None,
        context_order_id: str | None,
    ) -> RouteDecision:
        text = normalize_text(message)
        risk = assess_risk(text)
        intent = detect_intent(text)
        order_id = extract_order_id(text, context_order_id)
        personal = is_personal_order_query(text)
        context_dependent = is_context_dependent(text)

        rewritten = text
        if context_dependent and previous_user_message:
            rewritten = f"{normalize_text(previous_user_message)} {text}"

        if risk.is_high_risk:
            return RouteDecision(
                channel="human",
                intent=intent,
                reason=risk.reason_code or "risk",
                risk=risk,
                order_id=order_id,
                context_dependent=context_dependent,
                rewritten_query=rewritten,
                personal_order=personal,
            )

        if personal and order_id and intent in {
            "order_status",
            "logistics",
            "refund_status",
            "shipping_time",
        }:
            return RouteDecision(
                channel="tool",
                intent=intent,
                reason="order_specific",
                risk=risk,
                order_id=order_id,
                context_dependent=context_dependent,
                rewritten_query=rewritten,
                personal_order=personal,
                tool_intent=_tool_intent(intent),
            )

        return RouteDecision(
            channel="knowledge",
            intent=intent,
            reason="knowledge_lookup",
            risk=risk,
            order_id=order_id,
            context_dependent=context_dependent,
            rewritten_query=rewritten,
            personal_order=personal,
        )
