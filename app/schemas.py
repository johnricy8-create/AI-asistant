from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatContext(BaseModel):
    order_id: str | None = None
    product_id: str | None = None
    locale: str = "zh-CN"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)
    user_id: str = "c99001"
    mode: Literal["copilot", "self_service", "shadow"] = "copilot"
    channel: str = "workbench"
    context: ChatContext = Field(default_factory=ChatContext)
    regenerate: bool = False
    trace_id: str | None = None

    model_config = ConfigDict(extra="ignore")


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    section: str
    url: str = ""
    score: float = 0.0
    snippet: str = ""


class SuggestedAction(BaseModel):
    type: str
    label: str


class RiskFlag(BaseModel):
    code: str
    label: str
    level: Literal["info", "warn", "danger"] = "warn"


class TraceStep(BaseModel):
    name: str
    label: str
    status: Literal["ok", "warn", "error"] = "ok"
    latency_ms: int = 0
    detail: str = ""


class OrderSnapshot(BaseModel):
    order_id: str
    product_name: str
    amount: float
    status: str
    status_detail: str
    paid_at: str | None = None
    shipped_at: str | None = None
    signed_at: str | None = None
    carrier: str | None = None
    tracking_no: str | None = None
    latest_event: str | None = None
    estimated_delivery: str | None = None
    refund_status: str | None = None
    price_protection_until: str | None = None


class FAQDebug(BaseModel):
    faq_id: str
    std_q: str
    score: float
    margin: float
    threshold: float
    passed: bool
    matched_question: str
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class PriorityInfo(BaseModel):
    level: Literal["P0", "P1", "P2", "P3"]
    label: str
    handling: Literal["auto_reply", "copilot", "fallback_human", "human"]
    handling_label: str
    reason: str
    source: str = "rule"
    confidence: float = 0.8
    need_human: bool = False
    auto_reply_eligible: bool = False
    product_issue: bool = False
    large_amount: bool = False
    money_issue: bool = False
    category: str = "general"
    category_label: str = "一般咨询"


class ChatDebug(BaseModel):
    normalized_query: str
    rewritten_query: str
    intent: str
    order_id: str | None = None
    context_dependent: bool = False
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    faq: FAQDebug | None = None
    retrieval: list[dict[str, Any]] = Field(default_factory=list)
    tool: dict[str, Any] | None = None
    llm_error: str | None = None
    model_mode: str = "mock"
    priority: PriorityInfo | None = None


class ChatResponse(BaseModel):
    message_id: str
    session_id: str
    reply: str
    channel: Literal["faq", "rag", "tool", "human"]
    confidence: float
    need_human: bool
    citations: list[Citation] = Field(default_factory=list)
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    order_snapshot: OrderSnapshot | None = None
    trace: list[TraceStep] = Field(default_factory=list)
    latency_ms: int = 0
    model: str = ""
    model_mode: str = "mock"
    priority: PriorityInfo | None = None
    auto_replied: bool = False
    acknowledgement: str | None = None
    disclaimer: str = "AI 生成内容，请客服核对后发送。"
    debug: ChatDebug | None = None
    handoff: dict[str, Any] | None = None


class FeedbackRequest(BaseModel):
    message_id: str
    type: Literal["helpful", "not_helpful", "wrong", "adopted", "edited", "policy_outdated"]
    reason_code: str | None = None
    comment: str | None = None
    edited_reply: str | None = None


class FeedbackResponse(BaseModel):
    ok: bool = True
    feedback_id: str


class HandoffRequest(BaseModel):
    conversation_id: str
    reason_code: str = "user_request"
    priority: Literal["normal", "high", "urgent"] = "normal"
    skill_group: str = "after_sales"
    context: dict[str, Any] = Field(default_factory=dict)


class HandoffResponse(BaseModel):
    handoff_id: str
    status: str = "queued"
    skill_group: str
    priority: str
    created_at: str
    context: dict[str, Any] = Field(default_factory=dict)


class Scenario(BaseModel):
    id: str
    title: str
    message: str
    order_id: str
    hint: str
    description: str
    seed_messages: list[dict[str, str]] = Field(default_factory=list)


class SeedRequest(BaseModel):
    session_id: str
    messages: list[dict[str, str]] = Field(default_factory=list)
