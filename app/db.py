from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    user_id: Mapped[str] = mapped_column(String(128), default="")
    message: Mapped[str] = mapped_column(Text, default="")
    reply: Mapped[str] = mapped_column(Text, default="")
    channel: Mapped[str] = mapped_column(String(32), default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    need_human: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str] = mapped_column(String(128), default="")
    model_mode: Mapped[str] = mapped_column(String(64), default="")
    trace_json: Mapped[str] = mapped_column(Text, default="[]")
    citations_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(String(64), index=True)
    feedback_type: Mapped[str] = mapped_column(String(64), default="")
    reason_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    edited_reply: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Handoff(Base):
    __tablename__ = "handoffs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    handoff_id: Mapped[str] = mapped_column(String(64), index=True)
    conversation_id: Mapped[str] = mapped_column(String(128), index=True)
    reason_code: Mapped[str] = mapped_column(String(128), default="")
    priority: Mapped[str] = mapped_column(String(32), default="normal")
    skill_group: Mapped[str] = mapped_column(String(64), default="after_sales")
    status: Mapped[str] = mapped_column(String(32), default="queued")
    context_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Database:
    """SQLite for the demo; swap DATABASE_URL to MySQL for production."""

    def __init__(self, database_url: str):
        if database_url.startswith("sqlite:///"):
            raw_path = database_url.replace("sqlite:///", "", 1)
            if raw_path and raw_path != ":memory:":
                Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=connect_args, future=True)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    def log_chat(
        self,
        *,
        message_id: str,
        session_id: str,
        user_id: str,
        message: str,
        reply: str,
        channel: str,
        confidence: float,
        need_human: bool,
        latency_ms: int,
        model: str,
        model_mode: str,
        trace: list[dict[str, Any]],
        citations: list[dict[str, Any]],
    ) -> int:
        with self.session_factory() as session:
            row = ChatLog(
                message_id=message_id,
                session_id=session_id,
                user_id=user_id,
                message=message,
                reply=reply,
                channel=channel,
                confidence=confidence,
                need_human=need_human,
                latency_ms=latency_ms,
                model=model,
                model_mode=model_mode,
                trace_json=json.dumps(trace, ensure_ascii=False),
                citations_json=json.dumps(citations, ensure_ascii=False),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def save_feedback(
        self,
        *,
        message_id: str,
        feedback_type: str,
        reason_code: str | None = None,
        comment: str | None = None,
        edited_reply: str | None = None,
    ) -> int:
        with self.session_factory() as session:
            row = Feedback(
                message_id=message_id,
                feedback_type=feedback_type,
                reason_code=reason_code,
                comment=comment,
                edited_reply=edited_reply,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def save_handoff(
        self,
        *,
        handoff_id: str,
        conversation_id: str,
        reason_code: str,
        priority: str,
        skill_group: str,
        context: dict[str, Any],
    ) -> int:
        with self.session_factory() as session:
            row = Handoff(
                handoff_id=handoff_id,
                conversation_id=conversation_id,
                reason_code=reason_code,
                priority=priority,
                skill_group=skill_group,
                context_json=json.dumps(context, ensure_ascii=False),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def stats(self) -> dict[str, int]:
        with self.session_factory() as session:
            chat_count = session.scalar(select(func.count()).select_from(ChatLog)) or 0
            feedback_count = session.scalar(select(func.count()).select_from(Feedback)) or 0
            handoff_count = session.scalar(select(func.count()).select_from(Handoff)) or 0
        return {
            "chat_logs": int(chat_count),
            "feedback": int(feedback_count),
            "handoffs": int(handoff_count),
        }
