from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import Settings, get_settings
from app.data.seed import FAQ_SEED, KB_SEED
from app.db import Database
from app.orchestration.orchestrator import ChatOrchestrator
from app.orchestration.session import SessionStore
from app.rag.faq import FAQIndex
from app.rag.generator import GeneratorService
from app.rag.index import HybridRetriever

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass
class AppServices:
    settings: Settings
    faq_index: FAQIndex
    retriever: HybridRetriever
    sessions: SessionStore
    generator: GeneratorService
    db: Database
    orchestrator: ChatOrchestrator


def build_services(settings: Settings) -> AppServices:
    faq_index = FAQIndex(FAQ_SEED, default_margin=settings.faq_margin_default)
    retriever = HybridRetriever(KB_SEED)
    sessions = SessionStore()
    generator = GeneratorService(settings)
    db = Database(settings.database_url)
    orchestrator = ChatOrchestrator(
        settings=settings,
        faq_index=faq_index,
        retriever=retriever,
        sessions=sessions,
        generator=generator,
        db=db,
    )
    return AppServices(
        settings=settings,
        faq_index=faq_index,
        retriever=retriever,
        sessions=sessions,
        generator=generator,
        db=db,
        orchestrator=orchestrator,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = build_services(settings)
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="优选商城 RAG 智能客服 Copilot Demo",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.svg")

    return app


app = create_app()
