from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration. All values can be overridden via .env."""

    app_name: str = "优选商城 · 智能客服 Copilot"
    app_version: str = "0.1.0-demo"
    env: str = "demo"

    database_url: str = "sqlite:///./data/app.db"

    zhipu_api_key: str = ""
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    zhipu_model: str = "glm-4.7-flash"
    zhipu_embedding_model: str = "embedding-3"
    zhipu_disable_thinking: bool = True
    zhipu_max_tokens: int = 800
    zhipu_timeout_seconds: float = 10.0
    zhipu_total_timeout_seconds: float = 8.0
    zhipu_max_retries: int = 2
    zhipu_retry_backoff_seconds: float = 1.0
    zhipu_cooldown_seconds: int = 20
    priority_total_timeout_seconds: float = 3.0
    llm_mode: str = "auto"  # auto | mock | zhipu
    request_timeout_seconds: float = 30.0

    faq_threshold_default: float = 0.82
    faq_margin_default: float = 0.03
    rag_min_score: float = 0.20
    priority_large_amount_threshold: float = 500.0
    max_context_chars: int = 6000
    max_history_turns: int = 6

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def zhipu_enabled(self) -> bool:
        return self.llm_mode in {"auto", "zhipu"} and bool(self.zhipu_api_key.strip())

    @property
    def model_mode(self) -> str:
        if self.zhipu_enabled:
            return "zhipu"
        return "mock"


def get_settings() -> Settings:
    return Settings()
