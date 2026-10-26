from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from app.config import Settings
from app.core.text import clamp, mask_pii, normalize_text
from app.data.orders import build_snapshot, get_order, query_tool
from app.db import Database
from app.orchestration.router import IntentRouter, RouteDecision, has_domain_signal
from app.orchestration.priority import PriorityAssessment, PriorityEngine
from app.orchestration.session import SessionStore
from app.rag.faq import FAQIndex, FAQMatch
from app.rag.generator import GeneratorService
from app.schemas import (
    ChatDebug,
    ChatRequest,
    ChatResponse,
    Citation,
    FAQDebug,
    OrderSnapshot,
    RiskFlag,
    SuggestedAction,
    TraceStep,
)

logger = logging.getLogger(__name__)

SMALLTALK = {"你好", "您好", "在吗", "hi", "hello", "嗨", "哈喽", "有人吗"}
SMALLTALK_REPLY = (
    "您好，我是优选商城智能客服助手。"
    "我可以帮您查订单、物流、退款进度，也可以解答退换货、运费、发票、预售等问题。"
    "您想先了解哪一项？"
)
ACKNOWLEDGEMENT = "您好，已收到您的问题，我先帮您查询一下，请稍候。"
FORBIDDEN_PROMISES = ["保证退款", "一定退款", "百分百", "肯定能退", "绝对可以", "一定赔偿"]
CHANNEL_LABELS = {
    "faq": "FAQ 通道",
    "rag": "RAG 通道",
    "tool": "业务工具通道",
    "human": "人工兜底",
    "knowledge": "知识检索",
}


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _citation_from_hit(hit: dict[str, Any]) -> Citation:
    return Citation(
        doc_id=hit["doc_id"],
        chunk_id=hit["chunk_id"],
        title=hit["title"],
        section=hit["section"],
        url=hit.get("source_uri", ""),
        score=float(hit.get("score", 0.0)),
        snippet=hit.get("snippet", ""),
    )


def _citation_from_faq(faq: FAQMatch) -> Citation:
    return Citation(
        doc_id=faq.faq_id,
        chunk_id=faq.faq_id,
        title=f"标准问答 · {faq.category}",
        section=faq.std_q,
        url=f"/faq/{faq.faq_id}",
        score=float(faq.score),
        snippet=faq.answer[:180],
    )


def _citations_from_hits(
    hits: list[dict[str, Any]],
    min_score: float = 0.35,
    limit: int = 3,
) -> list[Citation]:
    filtered = [hit for hit in hits if float(hit.get("score", 0.0)) >= min_score]
    if not filtered and hits:
        filtered = hits[:1]
    return [_citation_from_hit(hit) for hit in filtered[:limit]]


def _risk_models(flags: list[dict[str, str]]) -> list[RiskFlag]:
    seen: set[str] = set()
    result: list[RiskFlag] = []
    for item in flags:
        code = item["code"]
        if code in seen:
            continue
        seen.add(code)
        result.append(RiskFlag(**item))
    return result


def _add_flag(
    flags: list[dict[str, str]],
    code: str,
    label: str,
    level: str = "warn",
) -> None:
    if any(item["code"] == code for item in flags):
        return
    flags.append({"code": code, "label": label, "level": level})


def _skill_group(reason_code: str | None) -> str:
    return {
        "refund_request": "after_sales",
        "price_protection": "after_sales",
        "complaint": "complaint",
        "address_change": "order_service",
        "order_cancel": "order_service",
        "logistics": "logistics",
        "invoice": "finance",
        "order_not_found": "order_service",
        "low_confidence": "after_sales",
        "out_of_scope": "after_sales",
        "user_request": "after_sales",
        "product_issue": "after_sales",
        "large_amount": "after_sales",
        "money_issue": "after_sales",
        "priority_high": "after_sales",
    }.get(reason_code or "", "after_sales")


def _priority_skill_group(priority: PriorityAssessment) -> str:
    return {
        "safety": "complaint",
        "complaint": "complaint",
        "product_quality": "after_sales",
        "large_amount": "after_sales",
        "money": "after_sales",
        "logistics": "logistics",
        "invoice": "finance",
        "address": "order_service",
        "general": "after_sales",
    }.get(priority.category, "after_sales")


def _priority_reason_code(
    priority: PriorityAssessment,
    decision: RouteDecision | None,
) -> str:
    if priority.category == "product_quality":
        return "product_issue"
    if priority.category == "large_amount":
        return "large_amount"
    if priority.category == "complaint" or priority.category == "safety":
        return "complaint"
    if priority.category == "logistics":
        return "logistics"
    if priority.category == "invoice":
        return "invoice"
    if priority.category == "address":
        return "address_change"
    if priority.category == "money":
        if decision and decision.risk.reason_code in {
            "price_protection",
            "refund_request",
        }:
            return decision.risk.reason_code
        return "money_issue"
    return "priority_high"


def _policy_query_for_reason(reason_code: str | None) -> str:
    return {
        "price_protection": "价保 退差价 申请条件 不适用",
        "refund_request": "退款 到账 原路退回",
        "complaint": "投诉 处理 客服 售后",
        "address_change": "修改地址 收货地址",
        "order_cancel": "取消订单 退款",
        "user_request": "客服 售后 处理",
        "low_confidence": "客服 售后",
        "out_of_scope": "客服 售后",
        "order_not_found": "订单查询 物流",
        "product_issue": "商品质量 破损 退换货 售后",
        "large_amount": "大额订单 退款 售后 人工",
        "money_issue": "退款 价保 定金 规则",
        "invoice": "发票 开票 红冲 电子发票",
        "logistics": "物流异常 签收 配送 丢件",
        "priority_high": "客服 售后 处理",
    }.get(reason_code or "", "客服 售后")


def _suggested_actions(
    channel: str,
    intent: str,
    has_order: bool,
) -> list[SuggestedAction]:
    actions: list[SuggestedAction] = []
    if channel == "faq":
        if intent == "return_policy":
            actions.append(SuggestedAction(type="open_return", label="去申请退货"))
        actions.append(SuggestedAction(type="open_policy", label="查看规则原文"))
    elif channel == "rag":
        actions.append(SuggestedAction(type="open_policy", label="查看政策原文"))
        actions.append(SuggestedAction(type="handoff", label="转人工核实"))
    elif channel == "tool":
        if intent == "refund_status":
            actions.append(SuggestedAction(type="view_refund", label="查看退款进度"))
        elif intent == "logistics":
            actions.append(SuggestedAction(type="query_logistics", label="查看物流"))
        else:
            actions.append(SuggestedAction(type="urge_shipping", label="催发货"))
        actions.append(SuggestedAction(type="handoff", label="转人工"))
    else:
        actions.append(SuggestedAction(type="handoff", label="转接售后专席"))
        if has_order:
            actions.append(SuggestedAction(type="view_order", label="查看订单"))
    if has_order and all(action.type != "view_order" for action in actions):
        actions.append(SuggestedAction(type="view_order", label="查看订单"))
    return actions[:3]


class ChatOrchestrator:
    def __init__(
        self,
        settings: Settings,
        faq_index: FAQIndex,
        retriever: Any,
        reranker: Any,
        sessions: SessionStore,
        generator: GeneratorService,
        db: Database,
    ):
        self.settings = settings
        self.faq_index = faq_index
        self.retriever = retriever
        self.reranker = reranker
        self.sessions = sessions
        self.generator = generator
        self.db = db
        self.router = IntentRouter()
        self.priority_engine = PriorityEngine(
            settings.priority_large_amount_threshold
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        trace: list[TraceStep] = []
        session = self.sessions.get_or_create(request.session_id)
        history = session.recent(self.settings.max_history_turns)
        previous_user = session.last_user_message

        if request.regenerate and previous_user:
            message = previous_user
        else:
            message = request.message.strip()
            self.sessions.append(
                request.session_id,
                "customer",
                message,
                {"order_id": request.context.order_id},
            )
            trace.append(
                TraceStep(
                    name="holding",
                    label="自动接待",
                    status="ok",
                    latency_ms=0,
                    detail=ACKNOWLEDGEMENT,
                )
            )

        normalize_start = time.perf_counter()
        normalized = normalize_text(message)
        trace.append(
            TraceStep(
                name="normalize",
                label="文本规范化",
                status="ok",
                latency_ms=_elapsed_ms(normalize_start),
                detail=f"规范化结果：{normalized[:80]}",
            )
        )

        # Small talk is handled as a safe FAQ-like shortcut.
        if normalized in SMALLTALK:
            faq = FAQMatch(
                faq_id="smalltalk",
                std_q="问候",
                answer=SMALLTALK_REPLY,
                category="通用",
                risk_level="low",
                score=1.0,
                margin=1.0,
                threshold=0.0,
                passed=True,
                matched_question=normalized,
            )
            priority = PriorityAssessment(
                level="P3",
                handling="auto_reply",
                reason="问候/闲聊，低风险",
                source="rule",
                confidence=0.99,
            )
            trace.append(
                TraceStep(
                    name="priority",
                    label="优先级审核",
                    status="ok",
                    latency_ms=0,
                    detail="P3 低优先 · 自动回复 · 问候/闲聊",
                )
            )
            auto_replied = self._auto_replied(
                request.mode, priority, "faq", 1.0, need_human=False
            )
            return await self._finish(
                request=request,
                started=started,
                trace=trace,
                faq=faq,
                channel="faq",
                confidence=1.0,
                hits=[],
                tool_result=None,
                order_snapshot=build_snapshot(get_order(request.context.order_id)),
                intent="smalltalk",
                normalized=normalized,
                rewritten=normalized,
                context_dependent=False,
                risk_flags=[],
                reply=SMALLTALK_REPLY,
                model="faq-standard",
                model_mode="standard",
                priority=priority,
                auto_replied=auto_replied,
            )

        route_start = time.perf_counter()
        decision = self.router.route(
            message=normalized,
            previous_user_message=previous_user,
            context_order_id=request.context.order_id,
        )
        trace.append(
            TraceStep(
                name="risk_check",
                label="风险预检",
                status="error" if decision.risk.is_high_risk else "ok",
                latency_ms=0,
                detail=(
                    "命中高风险规则："
                    + "、".join(flag["label"] for flag in decision.risk.flags)
                    if decision.risk.is_high_risk
                    else "未命中高风险规则"
                ),
            )
        )
        trace.append(
            TraceStep(
                name="route",
                label="意图路由",
                status="warn" if decision.channel == "human" else "ok",
                latency_ms=_elapsed_ms(route_start),
                detail=(
                    f"{CHANNEL_LABELS.get(decision.channel, decision.channel)}"
                    f" · 意图 {decision.intent} · 原因 {decision.reason}"
                ),
            )
        )

        order_id = decision.order_id or request.context.order_id
        order_snapshot = build_snapshot(get_order(order_id))
        risk_flags: list[dict[str, str]] = [*decision.risk.flags]
        citations: list[Citation] = []
        hits: list[dict[str, Any]] = []
        tool_result: dict[str, Any] | None = None
        faq: FAQMatch | None = None
        channel = decision.channel
        confidence = 0.0
        reply = ""

        priority_start = time.perf_counter()
        priority = await self.priority_engine.assess(
            question=normalized,
            order_snapshot=order_snapshot,
            risk_flags=decision.risk.flags,
            intent=decision.intent,
            personal_order=decision.personal_order,
            classifier=self.generator.classify_priority,
        )
        trace.append(
            TraceStep(
                name="priority",
                label="优先级审核",
                status="error" if priority.level in {"P0", "P1"} else "ok",
                latency_ms=_elapsed_ms(priority_start),
                detail=(
                    f"{priority.label} · {priority.handling_label} "
                    f"· {priority.reason} · 来源 {priority.source}"
                ),
            )
        )
        if priority.product_issue:
            _add_flag(risk_flags, "product_issue", "商品质量问题", "danger")
        if priority.large_amount:
            _add_flag(
                risk_flags,
                "large_amount",
                f"大额订单（≥¥{self.settings.priority_large_amount_threshold:.0f}）",
                "danger",
            )
        if priority.money_issue:
            _add_flag(risk_flags, "money", "涉及退款/差价", "danger")
        if priority.level == "P0":
            _add_flag(risk_flags, "priority_p0", "P0 紧急问题", "danger")

        if priority.handling in {"human", "fallback_human"}:
            channel = "human"
            hits = self._retrieve_policy(
                self._priority_policy_query(priority, decision.intent), trace
            )
            citations = _citations_from_hits(hits)
            reply = self._priority_fallback_reply(priority)
            confidence = 0.0
        elif decision.channel == "human":
            channel = "human"
            hits = self._retrieve_policy(
                _policy_query_for_reason(decision.reason), trace
            )
            citations = [_citation_from_hit(hit) for hit in hits[:3]]
            reply = self._human_reply(decision.reason)
            confidence = 0.0
        elif decision.channel == "tool":
            tool_start = time.perf_counter()
            tool_result = query_tool(decision.tool_intent or "order_status", order_id)
            trace.append(
                TraceStep(
                    name="tool",
                    label="订单/物流工具",
                    status="ok" if tool_result.get("found") else "error",
                    latency_ms=_elapsed_ms(tool_start),
                    detail=tool_result.get("detail", ""),
                )
            )
            if not tool_result.get("found"):
                channel = "human"
                _add_flag(risk_flags, "order_not_found", "未查询到订单", "danger")
                priority = self._force_human_priority(priority, "未查询到订单")
                hits = self._retrieve_policy(
                    _policy_query_for_reason("order_not_found"), trace
                )
                citations = [_citation_from_hit(hit) for hit in hits[:3]]
                reply = tool_result["answer"]
                confidence = 0.0
            else:
                channel = "tool"
                hits = self._retrieve_policy(tool_result.get("policy_query", ""), trace)
                citations = [_citation_from_hit(hit) for hit in hits[:3]]
                confidence = 0.95
        else:
            faq_start = time.perf_counter()
            faq = self.faq_index.search(decision.rewritten_query)
            trace.append(
                TraceStep(
                    name="faq_match",
                    label="FAQ 语义匹配",
                    status="ok" if faq and faq.passed else "warn",
                    latency_ms=_elapsed_ms(faq_start),
                    detail=(
                        f"{faq.faq_id} · 分数 {faq.score:.3f} · 分差 {faq.margin:.3f} "
                        f"· 阈值 {faq.threshold:.2f}"
                        if faq and faq.passed
                        else (
                            f"未命中 · 最高 {faq.score:.3f} · 阈值 {faq.threshold:.2f}"
                            if faq
                            else "无候选"
                        )
                    ),
                )
            )
            if faq and faq.passed:
                channel = "faq"
                citations = [_citation_from_faq(faq)]
                confidence = faq.score
            elif decision.intent == "general" and not has_domain_signal(
                decision.rewritten_query
            ):
                channel = "human"
                _add_flag(risk_flags, "out_of_scope", "问题超出商城知识域", "warn")
                priority = self._force_human_priority(priority, "超出商城知识域")
                trace.append(
                    TraceStep(
                        name="scope_check",
                        label="知识域判断",
                        status="warn",
                        latency_ms=0,
                        detail="未命中商城知识域，转人工兜底",
                    )
                )
                reply = (
                    "抱歉，我暂时只能处理订单、物流、退换货、运费、发票、预售等商城相关问题。"
                    "您可以换个问法，或我帮您转人工客服。"
                )
                confidence = 0.0
            else:
                hits = self._search_and_rerank(
                    decision.rewritten_query,
                    trace,
                    top_k=self.settings.recall_top_k,
                    rerank_top_k=self.settings.rerank_top_k,
                )
                top_score = hits[0]["score"] if hits else 0.0
                min_score = (
                    self.settings.rerank_min_score
                    if getattr(self.reranker, "active", False)
                    else self.settings.rag_min_score
                )
                trace.append(
                    TraceStep(
                        name="threshold",
                        label="相关性阈值",
                        status="ok" if top_score >= min_score else "warn",
                        latency_ms=0,
                        detail=f"最高分 {top_score:.3f} · 阈值 {min_score:.2f}",
                    )
                )
                if not hits or top_score < min_score:
                    channel = "human"
                    _add_flag(risk_flags, "low_confidence", "检索置信度不足", "warn")
                    priority = self._force_human_priority(priority, "检索置信度不足")
                    citations = [_citation_from_hit(hit) for hit in hits[:2]]
                    reply = (
                        "抱歉，我暂时没有检索到足够明确的规则，为避免给您错误信息，"
                        "建议转人工客服进一步核实。"
                    )
                    confidence = 0.0
                else:
                    channel = "rag"
                    citations = [_citation_from_hit(hit) for hit in hits[:3]]
                    confidence = clamp(0.45 + top_score * 0.65, 0.0, 0.94)

        generation_start = time.perf_counter()
        if not reply:
            generated = await self.generator.generate(
                channel=channel,
                question=normalized,
                faq=faq,
                hits=hits,
                tool=tool_result,
                order_snapshot=order_snapshot,
                history=history,
            )
            reply = generated.text
            model = generated.model
            model_mode = generated.mode
            if generated.error:
                error_text = generated.error
                if any(
                    token in error_text
                    for token in ["429", "限流", "冷却", "超时", "Timeout", "暂不可用"]
                ):
                    _add_flag(
                        risk_flags,
                        "llm_rate_limited",
                        "模型限流/超时，已降级模板",
                        "warn",
                    )
                else:
                    _add_flag(
                        risk_flags,
                        "llm_fallback",
                        "大模型调用失败，已降级模板",
                        "warn",
                    )
        else:
            model = "handoff-template" if channel == "human" else "faq-standard"
            model_mode = "standard"
            generated = None

        if channel == "rag" and generated and generated.mode == "mock_fallback":
            confidence = clamp(confidence - 0.05, 0.0, 1.0)

        need_human = channel == "human"
        auto_replied = self._auto_replied(
            request.mode, priority, channel, confidence, need_human
        )
        if auto_replied:
            trace.append(
                TraceStep(
                    name="auto_reply",
                    label="低优先自动回复",
                    status="ok",
                    latency_ms=0,
                    detail="P3 低优先且置信度达标，已满足自动回复条件",
                )
            )

        trace.append(
            TraceStep(
                name="generate",
                label="生成建议话术",
                status="warn" if generated and generated.mode == "mock_fallback" else "ok",
                latency_ms=_elapsed_ms(generation_start),
                detail=f"{model} · {model_mode}",
            )
        )

        guard_start = time.perf_counter()
        if mask_pii(reply) != reply:
            _add_flag(risk_flags, "pii_in_reply", "回复中可能包含个人信息", "warn")
        if any(word in reply for word in FORBIDDEN_PROMISES):
            _add_flag(risk_flags, "policy_risk", "可能存在超政策承诺", "danger")
        trace.append(
            TraceStep(
                name="guard",
                label="输出风控",
                status="warn" if risk_flags else "ok",
                latency_ms=_elapsed_ms(guard_start),
                detail=(
                    "、".join(flag["label"] for flag in risk_flags)
                    if risk_flags
                    else "通过"
                ),
            )
        )

        return await self._finish(
            request=request,
            started=started,
            trace=trace,
            faq=faq,
            channel=channel,
            confidence=confidence,
            hits=hits,
            tool_result=tool_result,
            order_snapshot=order_snapshot,
            intent=decision.intent,
            normalized=normalized,
            rewritten=decision.rewritten_query,
            context_dependent=decision.context_dependent,
            risk_flags=risk_flags,
            reply=reply,
            model=model,
            model_mode=model_mode,
            decision=decision,
            llm_error=generated.error if generated else None,
            priority=priority,
            auto_replied=auto_replied,
        )

    def _priority_policy_query(
        self, priority: PriorityAssessment, intent: str
    ) -> str:
        return {
            "product_quality": "商品质量 破损 退换货 售后",
            "large_amount": "大额订单 退款 售后 人工",
            "money": "退款 价保 定金 差价 规则",
            "logistics": "物流异常 签收 配送 丢件",
            "invoice": "发票 开票 红冲 电子发票",
            "address": "修改地址 收货地址",
            "complaint": "投诉 处理 客服 售后",
            "safety": "投诉 处理 安全 人工",
            "general": "客服 售后 处理",
        }.get(priority.category, "客服 售后 处理")

    def _priority_fallback_reply(self, priority: PriorityAssessment) -> str:
        category = priority.category
        if priority.level == "P0" or category == "safety":
            return (
                "您好，您反馈的问题我们非常重视，已优先转接人工客服处理。"
                "请您保持通讯畅通，具体以平台核实结果为准。"
            )
        if category == "product_quality":
            return (
                "您好，非常抱歉给您带来不便。为保障您的权益，建议您保留商品、包装及相关凭证，"
                "我已为您转接售后专席，后续会按平台售后规则为您核实处理。"
            )
        if category == "large_amount":
            return (
                "您好，这笔订单金额较大，为保障您的资金安全，退款/售后需要人工进一步核实，"
                "具体以平台审核结果为准。"
            )
        if category == "money":
            return (
                "您好，您反馈的差价/退款问题需要结合订单、活动规则和支付信息进一步核实，"
                "具体以平台规则和审核结果为准。我已为您转接售后专席，请稍候。"
            )
        if category == "logistics":
            return (
                "您好，已为您转接物流专席核实最新节点。"
                "如包裹破损或丢失，建议您保留物流凭证，我们会按平台规则跟进。"
            )
        if category == "invoice":
            return (
                "您好，发票开具/红冲需要财务专席核实，具体以税务和平台规则为准。"
                "我已为您转接财务专席，请稍候。"
            )
        if category == "address":
            return (
                "您好，修改地址涉及订单履约，建议由人工核实后再操作，以免包裹丢失。"
                "我已为您转接订单服务专席，请稍候。"
            )
        if category == "complaint":
            return (
                "非常抱歉给您带来不好的体验，我们已转接投诉专席优先跟进，"
                "后续会按平台服务规范处理。"
            )
        return (
            "您好，这个问题需要人工客服进一步核实，具体以平台规则和审核结果为准。"
            "我已为您转接售后专席，请稍候。"
        )

    def _auto_replied(
        self,
        mode: str,
        priority: PriorityAssessment | None,
        channel: str,
        confidence: float,
        need_human: bool,
    ) -> bool:
        if mode != "self_service":
            return False
        if priority is None or priority.handling != "auto_reply":
            return False
        if need_human:
            return False
        if channel not in {"faq", "rag", "tool"}:
            return False
        return confidence >= 0.6

    def _force_human_priority(
        self, priority: PriorityAssessment, reason_suffix: str
    ) -> PriorityAssessment:
        """Keep the priority level but reflect the final human handoff."""
        return PriorityAssessment(
            level=priority.level,
            handling="human",
            reason=f"{priority.reason}；{reason_suffix}",
            source=priority.source,
            confidence=priority.confidence,
            category=priority.category,
            product_issue=priority.product_issue,
            large_amount=priority.large_amount,
            money_issue=priority.money_issue,
            money_action=priority.money_action,
            complaint=priority.complaint,
            explicit_human=priority.explicit_human,
            fallback_keyword=priority.fallback_keyword,
        )

    def _search_and_rerank(
        self,
        query: str,
        trace: list[TraceStep],
        *,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not query:
            return []
        recall_k = top_k or self.settings.recall_top_k
        rerank_k = rerank_top_k or self.settings.rerank_top_k
        retriever_name = getattr(self.retriever, "name", "检索")
        reranker_name = getattr(self.reranker, "name", "重排")

        start = time.perf_counter()
        hits = self.retriever.search(query, top_k=recall_k)
        trace.append(
            TraceStep(
                name="retrieve",
                label=f"{retriever_name} 召回",
                status="ok" if hits else "warn",
                latency_ms=_elapsed_ms(start),
                detail=f"召回 {len(hits)} 条",
            )
        )

        if not hits:
            return []

        if getattr(self.reranker, "active", False):
            rerank_start = time.perf_counter()
            try:
                hits = self.reranker.rerank(query, hits, top_k=rerank_k)
                trace.append(
                    TraceStep(
                        name="rerank",
                        label=f"{reranker_name} 重排",
                        status="ok" if hits else "warn",
                        latency_ms=_elapsed_ms(rerank_start),
                        detail=(
                            f"重排 {len(hits)} 条 · 最高分 {hits[0]['score']:.3f}"
                            if hits
                            else "重排后无结果"
                        ),
                    )
                )
            except Exception:  # noqa: BLE001 - retrieval must degrade gracefully
                logger.exception("rerank failed; keeping retrieval order")
                hits = hits[:rerank_k]
                trace.append(
                    TraceStep(
                        name="rerank",
                        label="重排降级",
                        status="warn",
                        latency_ms=_elapsed_ms(rerank_start),
                        detail="重排失败，保留检索排序",
                    )
                )
        else:
            hits = hits[:rerank_k]
        return hits

    def _retrieve_policy(self, query: str, trace: list[TraceStep]) -> list[dict[str, Any]]:
        return self._search_and_rerank(
            query,
            trace,
            top_k=self.settings.recall_top_k,
            rerank_top_k=3,
        )

    def _human_reply(self, reason_code: str | None) -> str:
        if reason_code == "low_confidence":
            return (
                "抱歉，我暂时没有检索到足够明确的规则，为避免给您错误信息，"
                "建议转人工客服进一步核实。"
            )
        if reason_code == "order_not_found":
            return (
                "我没有查到这笔订单，可能是订单号有误或超出了当前查询范围。"
                "请您核对订单号，或我帮您转人工进一步查询。"
            )
        if reason_code == "complaint":
            return (
                "非常抱歉给您带来不好的体验。我已将您的问题和订单信息转接给投诉处理专席，"
                "会有专人优先跟进，请稍候。"
            )
        return (
            "您好，这个问题涉及退款、差价或投诉等需要人工核实的操作。"
            "我已为您转接售后专席，并把订单信息和问题摘要一起带过去了，请稍候。"
        )

    async def _finish(
        self,
        *,
        request: ChatRequest,
        started: float,
        trace: list[TraceStep],
        faq: FAQMatch | None,
        channel: str,
        confidence: float,
        hits: list[dict[str, Any]],
        tool_result: dict[str, Any] | None,
        order_snapshot: dict[str, Any] | None,
        intent: str,
        normalized: str,
        rewritten: str,
        context_dependent: bool,
        risk_flags: list[dict[str, str]],
        reply: str,
        model: str,
        model_mode: str,
        decision: RouteDecision | None = None,
        llm_error: str | None = None,
        priority: PriorityAssessment | None = None,
        auto_replied: bool = False,
    ) -> ChatResponse:
        message_id = f"m-{uuid.uuid4().hex[:12]}"
        need_human = channel == "human"
        risk_models = _risk_models(risk_flags)
        citations = []
        if channel == "faq" and faq is not None and faq.passed:
            citations = [_citation_from_faq(faq)]
        elif hits:
            citations = _citations_from_hits(
                hits,
                min_score=0.0 if getattr(self.reranker, "active", False) else 0.35,
            )

        priority_schema = priority.to_schema() if priority else None
        handoff = None
        if need_human:
            if priority and priority.handling in {"human", "fallback_human"}:
                reason_code = _priority_reason_code(priority, decision)
            else:
                reason_code = (
                    decision.risk.reason_code
                    if decision and decision.risk.reason_code
                    else (
                        "low_confidence"
                        if any(f["code"] == "low_confidence" for f in risk_flags)
                        else "user_request"
                    )
                )
            if any(f["code"] == "order_not_found" for f in risk_flags):
                reason_code = "order_not_found"
            if any(f["code"] == "out_of_scope" for f in risk_flags):
                reason_code = "out_of_scope"
            if (
                any(f["code"] == "low_confidence" for f in risk_flags)
                and reason_code == "priority_high"
            ):
                reason_code = "low_confidence"

            if priority_schema and priority_schema.level == "P0":
                handoff_priority = "urgent"
            elif priority_schema and priority_schema.level == "P1":
                handoff_priority = "high"
            elif reason_code == "complaint":
                handoff_priority = "urgent"
            elif any(f["level"] == "danger" for f in risk_flags):
                handoff_priority = "high"
            else:
                handoff_priority = "normal"

            handoff = {
                "reason_code": reason_code,
                "priority": handoff_priority,
                "skill_group": (
                    _priority_skill_group(priority)
                    if priority and priority.handling in {"human", "fallback_human"}
                    else _skill_group(reason_code)
                ),
                "summary": (
                    f"顾客问题：{normalized}；意图：{intent}；"
                    f"订单：{order_snapshot['order_id'] if order_snapshot else '无'}；"
                    f"优先级：{priority_schema.label if priority_schema else '未判定'}；"
                    f"类目：{priority_schema.category_label if priority_schema else '未分类'}；"
                    f"风险：{'、'.join(f['label'] for f in risk_flags) or '无'}"
                ),
                "order_snapshot": order_snapshot,
                "citations": [citation.model_dump() for citation in citations],
                "draft_reply": reply,
                "risk_flags": [flag.model_dump() for flag in risk_models],
                "priority_info": priority_schema.model_dump() if priority_schema else None,
                "auto_replied": auto_replied,
                "intent": intent,
                "rewritten_query": rewritten,
            }

        debug = ChatDebug(
            normalized_query=normalized,
            rewritten_query=rewritten,
            intent=intent,
            order_id=order_snapshot["order_id"] if order_snapshot else None,
            context_dependent=context_dependent,
            risk_flags=risk_models,
            faq=(
                FAQDebug(
                    faq_id=faq.faq_id,
                    std_q=faq.std_q,
                    score=round(faq.score, 4),
                    margin=round(faq.margin, 4),
                    threshold=faq.threshold,
                    passed=faq.passed,
                    matched_question=faq.matched_question,
                    candidates=faq.candidates,
                )
                if faq
                else None
            ),
            retrieval=hits,
            tool=tool_result,
            llm_error=llm_error,
            model_mode=model_mode,
            priority=priority_schema,
        )

        latency_ms = _elapsed_ms(started)
        response = ChatResponse(
            message_id=message_id,
            session_id=request.session_id,
            reply=reply,
            channel=channel,  # type: ignore[arg-type]
            confidence=round(confidence, 3),
            need_human=need_human,
            citations=citations,
            suggested_actions=_suggested_actions(
                channel,
                intent,
                has_order=order_snapshot is not None,
            ),
            risk_flags=risk_models,
            order_snapshot=OrderSnapshot(**order_snapshot) if order_snapshot else None,
            trace=trace,
            latency_ms=latency_ms,
            model=model,
            model_mode=model_mode,
            priority=priority_schema,
            auto_replied=auto_replied,
            acknowledgement=None if request.regenerate else ACKNOWLEDGEMENT,
            disclaimer=(
                "低优先级问题已自动回复；如顾客追问或涉及售后操作，请转人工。"
                if auto_replied
                else "AI 生成内容，请客服核对后发送。"
            ),
            debug=debug,
            handoff=handoff,
        )

        self.sessions.append(
            request.session_id,
            "assistant",
            reply,
            {
                "channel": channel,
                "confidence": confidence,
                "priority": priority_schema.level if priority_schema else None,
                "auto_replied": auto_replied,
            },
        )
        try:
            self.db.log_chat(
                message_id=message_id,
                session_id=request.session_id,
                user_id=request.user_id,
                message=normalized,
                reply=reply,
                channel=channel,
                confidence=round(confidence, 3),
                need_human=need_human,
                latency_ms=latency_ms,
                model=model,
                model_mode=model_mode,
                trace=[step.model_dump() for step in trace],
                citations=[citation.model_dump() for citation in citations],
            )
        except Exception:  # noqa: BLE001 - logging must never break the demo
            logger.exception("failed to persist chat log")

        return response
