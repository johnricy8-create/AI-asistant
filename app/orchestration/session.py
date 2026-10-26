from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class SessionState:
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)
    last_user_message: str | None = None
    last_channel: str | None = None
    created_at: str = field(default_factory=_now)

    def append(self, role: str, content: str, meta: dict[str, Any] | None = None) -> None:
        self.messages.append(
            {
                "role": role,
                "content": content,
                "meta": meta or {},
                "created_at": _now(),
            }
        )
        if role == "customer":
            self.last_user_message = content

    def recent(self, limit: int = 6) -> list[dict[str, Any]]:
        return self.messages[-limit:]


class SessionStore:
    """In-memory session store for the demo. Replace with Redis in production."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> SessionState:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionState(session_id=session_id)
                self._sessions[session_id] = session
            return session

    def seed(self, session_id: str, messages: list[dict[str, str]]) -> SessionState:
        session = self.get_or_create(session_id)
        with self._lock:
            session.messages = []
            session.slots = {}
            session.last_user_message = None
            for item in messages:
                role = item.get("role", "customer")
                content = item.get("content", "")
                if not content:
                    continue
                session.append(role, content, {"seeded": True})
        return session

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> SessionState:
        session = self.get_or_create(session_id)
        with self._lock:
            session.append(role, content, meta)
        return session

    def set_slot(self, session_id: str, key: str, value: Any) -> None:
        session = self.get_or_create(session_id)
        with self._lock:
            session.slots[key] = value

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def reset_all(self) -> None:
        with self._lock:
            self._sessions.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)
