from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Request

from app.data.orders import build_snapshot, list_orders
from app.data.seed import SCENARIOS
from app.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    HandoffRequest,
    HandoffResponse,
    SeedRequest,
)

router = APIRouter(prefix="/api/v1")


def _services(request: Request):
    return request.app.state.services


@router.get("/health")
def health(request: Request) -> dict:
    services = _services(request)
    settings = services.settings
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "model": settings.zhipu_model,
        "model_mode": settings.model_mode,
        "llm_mode": settings.llm_mode,
        "zhipu_configured": bool(settings.zhipu_api_key.strip()),
        "faq_count": len(services.faq_index.entries),
        "kb_count": len(services.retriever.chunks),
        "retrieval_backend": getattr(services.retriever, "name", "unknown"),
        "embedding_model": getattr(services.retriever, "embedding_model", "none"),
        "reranker": getattr(services.reranker, "name", "None"),
        "rerank_active": getattr(services.reranker, "active", False),
        "recall_top_k": settings.recall_top_k,
        "rerank_top_k": settings.rerank_top_k,
        "siliconflow_configured": settings.siliconflow_enabled,
        "sessions": services.sessions.count(),
        "stats": services.db.stats(),
    }


@router.get("/demo/scenarios")
def scenarios() -> list[dict]:
    return SCENARIOS


@router.get("/demo/orders")
def orders() -> list[dict]:
    return [build_snapshot(order) for order in list_orders()]


@router.post("/demo/seed")
def seed(payload: SeedRequest, request: Request) -> dict:
    services = _services(request)
    services.sessions.seed(payload.session_id, payload.messages)
    return {"ok": True, "session_id": payload.session_id, "messages": len(payload.messages)}


@router.post("/demo/reset")
def reset(request: Request) -> dict:
    services = _services(request)
    services.sessions.reset_all()
    return {"ok": True}


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    services = _services(request)
    return await services.orchestrator.chat(payload)


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest, request: Request) -> FeedbackResponse:
    services = _services(request)
    feedback_id = services.db.save_feedback(
        message_id=payload.message_id,
        feedback_type=payload.type,
        reason_code=payload.reason_code,
        comment=payload.comment,
        edited_reply=payload.edited_reply,
    )
    return FeedbackResponse(feedback_id=f"fb-{feedback_id}")


@router.post("/handoff", response_model=HandoffResponse)
def handoff(payload: HandoffRequest, request: Request) -> HandoffResponse:
    services = _services(request)
    handoff_id = f"ho-{uuid4().hex[:10]}"
    created_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    services.db.save_handoff(
        handoff_id=handoff_id,
        conversation_id=payload.conversation_id,
        reason_code=payload.reason_code,
        priority=payload.priority,
        skill_group=payload.skill_group,
        context=payload.context,
    )
    return HandoffResponse(
        handoff_id=handoff_id,
        status="queued",
        skill_group=payload.skill_group,
        priority=payload.priority,
        created_at=created_at,
        context=payload.context,
    )
