"""FastAPI-приложение hidden-patterns-combat backend."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.analysis.router import router as analysis_router
from app.auth.router import router as auth_router
from app.config import Settings, get_settings
from app.db.session import init_schema
from app.sources.router import router as sources_router


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        init_schema(settings)
        settings.storage_root.mkdir(parents=True, exist_ok=True)
        yield

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "auth", "description": "Регистрация и вход."},
            {"name": "sources", "description": "Excel-источники пользователя."},
            {"name": "analysis", "description": "Запуск и получение результатов анализа."},
            {"name": "health", "description": "Технические эндпоинты."},
        ],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok", "app": settings.app_name}

    app.include_router(auth_router, prefix="/api")
    app.include_router(sources_router, prefix="/api")
    app.include_router(analysis_router, prefix="/api")

    return app


app = create_app()
