from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.core.text import contains_any, normalize_text
from app.schemas import PriorityInfo

PRIORITY_ORDER = {"P3": 0, "P2": 1, "P1": 2, "P0": 3}
PRIORITY_LABELS = {
    "P0": "P0 紧急",
    "P1": "P1 高优先",
    "P2": "P2 中优先",
    "P3": "P3 低优先",
}
HANDLING_LABELS = {
    "auto_reply": "自动回复",
    "copilot": "建议人工复核",
    "fallback_human": "兜底 + 转人工",
    "human": "立即转人工",
}
CATEGORY_LABELS = {
    "safety": "安全/法律",
    "product_quality": "商品质量",
    "money": "资金/差价",
    "large_amount": "大额订单",
    "logistics": "物流",
    "invoice": "发票",
    "address": "地址/订单",
    "complaint": "投诉/舆情",
    "general": "一般咨询",
}

# LLM category names -> internal category names.
CATEGORY_MAP = {
    "资金安全": "money",
    "商品质量": "product_quality",
    "退款售后": "money",
    "物流": "logistics",
    "发票": "invoice",
    "账号": "general",
    "一般咨询": "general",
    "其他": "general",
}

# P0: safety, fraud, legal, public-opinion, account takeover.
P0_WORDS = [
    "人身安全",
    "受伤",
    "过敏",
    "中毒",
    "爆炸",
    "漏电",
    "起火",
    "食品安全",
    "变质",
    "假货",
    "假冒",
    "诈骗",
    "被骗",
    "盗刷",
    "账户被盗",
    "起诉",
    "律师",
    "12315",
    "曝光",
    "媒体",
    "监管",
    "报警",
    "生命危险",
    "自残",
]

PRODUCT_WORDS = [
    "质量",
    "质量有问题",
    "质量问题",
    "坏了",
    "商品坏了",
    "破损",
    "损坏",
    "瑕疵",
    "故障",
    "不能用",
    "无法使用",
    "不好用",
    "用不了",
    "失灵",
    "漏发",
    "错发",
    "少发",
    "发错",
    "货不对板",
    "假货",
    "过期",
    "变质",
    "异味",
    "不安全",
    "过敏",
    "跑毛",
    "漏水",
    "开裂",
]

# The user explicitly asked these to fall back even if the amount is small.
FALLBACK_KEYWORDS = [
    "商品坏了",
    "坏了",
    "差价",
    "质量有问题",
    "质量问题",
    "有瑕疵",
    "破损",
    "损坏",
    "漏发",
    "错发",
    "假货",
    "过期",
    "变质",
    "不安全",
    "不能用",
    "故障",
]

MONEY_WORDS = [
    "退款",
    "退钱",
    "退差价",
    "差价",
    "退定金",
    "赔偿",
    "补偿",
    "价保",
    "定金",
    "尾款",
]

MONEY_ACTION_PREFIXES = [
    "我要",
    "帮我",
    "给我",
    "麻烦",
    "请",
    "直接",
    "马上",
    "赶紧",
    "现在",
    "申请",
    "办理",
]

COMPLAINT_WORDS = [
    "投诉",
    "差评",
    "曝光",
    "12315",
    "起诉",
    "律师",
    "媒体",
    "监管",
    "报警",
    "骗子",
    "气死",
]

HUMAN_WORDS = ["转人工", "人工客服", "找人工", "真人", "人工服务", "我要人工"]
STATUS_WORDS = ["到账", "进度", "到哪", "签收", "运单", "物流", "配送", "派送"]

P2_WORDS = [
    "物流",
    "快递",
    "运单",
    "签收",
    "派送",
    "到货",
    "退款",
    "退货",
    "退换",
    "售后",
    "发票",
    "开票",
    "订单",
    "改地址",
    "修改地址",
    "催单",
    "催发货",
    "价保",
    "预售",
    "尾款",
    "会员",
    "优惠券",
    "支付",
    "分期",
    "配送",
]

P2_INTENTS = {
    "logistics",
    "refund_status",
    "invoice",
    "order_status",
    "return_policy",
    "presale_deposit",
    "address_change",
}


@dataclass
class PriorityAssessment:
    level: str
    handling: str
    reason: str
    source: str = "rule"
    confidence: float = 0.8
    category: str = "general"
    product_issue: bool = False
    large_amount: bool = False
    money_issue: bool = False
    money_action: bool = False
    complaint: bool = False
    explicit_human: bool = False
    fallback_keyword: str | None = None

    @property
    def label(self) -> str:
        return PRIORITY_LABELS[self.level]

    @property
    def handling_label(self) -> str:
        return HANDLING_LABELS[self.handling]

    @property
    def category_label(self) -> str:
        return CATEGORY_LABELS.get(self.category, CATEGORY_LABELS["general"])

    @property
    def need_human(self) -> bool:
        return self.handling in {"human", "fallback_human"}

    @property
    def auto_reply_eligible(self) -> bool:
        return self.handling == "auto_reply"

    def to_schema(self) -> PriorityInfo:
        return PriorityInfo(
            level=self.level,  # type: ignore[arg-type]
            label=self.label,
            handling=self.handling,  # type: ignore[arg-type]
            handling_label=self.handling_label,
            reason=self.reason,
            source=self.source,
            confidence=round(self.confidence, 2),
            need_human=self.need_human,
            auto_reply_eligible=self.auto_reply_eligible,
            product_issue=self.product_issue,
            large_amount=self.large_amount,
            money_issue=self.money_issue,
            category=self.category,
            category_label=self.category_label,
        )


Classifier = Callable[..., Awaitable[dict[str, Any] | None]]


class PriorityEngine:
    """Rule floor + LLM decision.

    Rules can raise priority and force a fallback. For non-forced P1/P2/P3
    questions, the LLM decides whether the answer can be auto-sent or needs a
    human. The LLM can never lower the rule's priority floor.
    """

    def __init__(self, large_amount_threshold: float = 500.0):
        self.large_amount_threshold = large_amount_threshold

    def _detect_money_action(self, text: str, status_query: bool) -> bool:
        # "差价" is a high-risk word per the business rule: always fall back.
        if "差价" in text:
            return True

        if not contains_any(text, MONEY_WORDS):
            return False
        if status_query and contains_any(text, ["查", "看", "到账", "进度", "到哪"]):
            return False
        starts_with_action = any(
            text.startswith(prefix) for prefix in MONEY_ACTION_PREFIXES
        )
        if not starts_with_action:
            return False
        # "我要退货" is a normal return, not a money write action.
        if contains_any(text, ["退货"]) and not contains_any(
            text, ["退款", "退差价", "差价", "退定金", "赔偿", "补偿"]
        ):
            return False
        return True

    def _category_for(
        self,
        *,
        text: str,
        intent: str,
        p0: bool,
        product_issue: bool,
        large_amount: bool,
        complaint: bool,
        money_issue: bool,
    ) -> str:
        if p0:
            return "safety"
        if product_issue:
            return "product_quality"
        if large_amount:
            return "large_amount"
        if complaint:
            return "complaint"
        if money_issue:
            return "money"
        if intent in {"logistics", "refund_status"} or contains_any(
            text, ["物流", "快递", "配送", "签收"]
        ):
            return "logistics"
        if intent == "invoice" or contains_any(text, ["发票", "开票"]):
            return "invoice"
        if intent == "address_change" or contains_any(text, ["地址"]):
            return "address"
        return "general"

    def _handling_for(
        self,
        level: str,
        *,
        product_issue: bool,
        large_amount: bool,
        money_action: bool,
        complaint: bool,
        explicit_human: bool,
        fallback_keyword: str | None,
    ) -> str:
        if level == "P0":
            return "human"
        if level == "P1":
            if (
                product_issue
                or large_amount
                or money_action
                or complaint
                or fallback_keyword
            ):
                return "fallback_human"
            return "copilot"
        if level == "P2":
            return "human" if explicit_human else "copilot"
        return "human" if explicit_human else "auto_reply"

    def _rules(
        self,
        *,
        question: str,
        order_snapshot: dict[str, Any] | None,
        intent: str,
        personal_order: bool,
    ) -> PriorityAssessment:
        text = normalize_text(question)
        amount = float((order_snapshot or {}).get("amount") or 0.0)
        product_issue = contains_any(text, PRODUCT_WORDS)
        fallback_keyword = next(
            (keyword for keyword in FALLBACK_KEYWORDS if keyword in text),
            None,
        )
        money_issue = contains_any(text, MONEY_WORDS)
        status_query = contains_any(text, STATUS_WORDS)
        money_action = self._detect_money_action(text, status_query)
        complaint = contains_any(text, COMPLAINT_WORDS)
        explicit_human = contains_any(text, HUMAN_WORDS)
        p0 = contains_any(text, P0_WORDS)

        if fallback_keyword == "差价":
            money_issue = True
            money_action = True
        if fallback_keyword and fallback_keyword != "差价":
            product_issue = True

        large_amount = amount >= self.large_amount_threshold and (
            product_issue
            or money_issue
            or complaint
            or fallback_keyword is not None
            or intent in P2_INTENTS
        )

        if p0:
            level = "P0"
            reason = "命中安全/诈骗/法律/舆情等紧急风险"
        elif product_issue:
            level = "P1"
            reason = (
                f"命中兜底关键词「{fallback_keyword}」，涉及商品问题"
                if fallback_keyword
                else "涉及商品质量问题"
            )
        elif large_amount:
            level = "P1"
            reason = (
                f"订单金额 ¥{amount:.0f} ≥ "
                f"{self.large_amount_threshold:.0f}，涉及售后"
            )
        elif money_action:
            level = "P1"
            reason = (
                f"命中兜底关键词「{fallback_keyword}」，涉及资金/差价问题"
                if fallback_keyword
                else "涉及退款/差价/赔偿等写操作"
            )
        elif complaint:
            level = "P1"
            reason = "涉及投诉/舆情风险"
        elif explicit_human:
            level = "P2"
            reason = "用户明确要求人工"
        elif money_issue and not status_query:
            level = "P1"
            reason = "涉及退款/定金/价保规则"
        elif status_query:
            level = "P2"
            reason = "订单/退款/物流状态查询"
        elif personal_order or contains_any(text, P2_WORDS) or intent in P2_INTENTS:
            level = "P2"
            reason = "常规售后/物流/发票咨询"
        else:
            level = "P3"
            reason = "一般规则/流程咨询"

        category = self._category_for(
            text=text,
            intent=intent,
            p0=p0,
            product_issue=product_issue,
            large_amount=large_amount,
            complaint=complaint,
            money_issue=money_issue,
        )
        handling = self._handling_for(
            level,
            product_issue=product_issue,
            large_amount=large_amount,
            money_action=money_action,
            complaint=complaint,
            explicit_human=explicit_human,
            fallback_keyword=fallback_keyword,
        )
        confidence = 0.9 if level in {"P0", "P1"} else 0.78
        return PriorityAssessment(
            level=level,
            handling=handling,
            reason=reason,
            source="rule",
            confidence=confidence,
            category=category,
            product_issue=product_issue,
            large_amount=large_amount,
            money_issue=money_issue,
            money_action=money_action,
            complaint=complaint,
            explicit_human=explicit_human,
            fallback_keyword=fallback_keyword,
        )

    async def assess(
        self,
        *,
        question: str,
        order_snapshot: dict[str, Any] | None,
        risk_flags: list[dict[str, str]],
        intent: str,
        personal_order: bool,
        classifier: Classifier | None = None,
    ) -> PriorityAssessment:
        rule = self._rules(
            question=question,
            order_snapshot=order_snapshot,
            intent=intent,
            personal_order=personal_order,
        )

        forced_fallback = (
            rule.product_issue
            or rule.large_amount
            or rule.money_action
            or rule.complaint
            or rule.fallback_keyword is not None
        )

        # P0 and forced fallbacks are authoritative; do not spend an LLM call
        # to try to lower them.
        if rule.level == "P0" or forced_fallback:
            return rule

        llm_data = None
        if classifier is not None:
            try:
                llm_data = await classifier(
                    question=question,
                    order_snapshot=order_snapshot,
                    risk_flags=risk_flags,
                    intent=intent,
                )
            except Exception:  # noqa: BLE001 - classification fallback must be safe
                llm_data = None

        if not llm_data:
            return rule

        raw_level = str(llm_data.get("priority") or "").upper()
        match = re.search(r"P[0-3]", raw_level)
        llm_level = match.group(0) if match else rule.level
        final_level = (
            llm_level
            if PRIORITY_ORDER.get(llm_level, 0) >= PRIORITY_ORDER[rule.level]
            else rule.level
        )

        product_issue = rule.product_issue or bool(llm_data.get("product_issue"))
        large_amount = rule.large_amount or bool(llm_data.get("large_amount"))
        money_action = rule.money_action or bool(llm_data.get("money_action"))
        money_issue = rule.money_issue or bool(llm_data.get("money_issue"))
        complaint = rule.complaint or bool(llm_data.get("complaint"))
        need_human = bool(llm_data.get("need_human"))
        auto_reply = bool(llm_data.get("auto_reply"))

        if product_issue or large_amount or money_action or complaint:
            forced_fallback = True
            if PRIORITY_ORDER[final_level] < PRIORITY_ORDER["P1"]:
                final_level = "P1"

        raw_category = str(llm_data.get("category") or "")
        llm_category = CATEGORY_MAP.get(raw_category, raw_category)
        category = (
            "product_quality"
            if product_issue
            else "large_amount"
            if large_amount
            else "complaint"
            if complaint
            else "money"
            if (money_issue or money_action)
            else llm_category
            if llm_category in CATEGORY_LABELS
            else rule.category
        )

        if final_level == "P0":
            handling = "human"
        elif forced_fallback:
            handling = "fallback_human"
        elif final_level == "P1":
            handling = "fallback_human" if need_human else "copilot"
        elif final_level == "P2":
            handling = "fallback_human" if need_human else "copilot"
        else:  # P3
            if need_human:
                handling = "fallback_human"
            elif auto_reply:
                handling = "auto_reply"
            else:
                handling = "copilot"

        reason = (
            rule.reason
            if forced_fallback
            else str(llm_data.get("reason") or rule.reason)
        )
        try:
            confidence = float(llm_data.get("confidence", 0.75))
        except (TypeError, ValueError):
            confidence = 0.75
        confidence = max(0.0, min(1.0, confidence))

        return PriorityAssessment(
            level=final_level,
            handling=handling,
            reason=reason,
            source="rule+llm",
            confidence=confidence,
            category=category,
            product_issue=product_issue,
            large_amount=large_amount,
            money_issue=money_issue,
            money_action=money_action,
            complaint=complaint,
            explicit_human=rule.explicit_human,
            fallback_keyword=rule.fallback_keyword,
        )
