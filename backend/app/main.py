"""FastAPI-приложение hidden-patterns-combat backend."""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager

# Uvicorn не трогает корневой логгер — поднимаем уровень, чтобы
# INFO-сообщения из app.* (в т.ч. app.mail с кодами) были видны
# в dev-консоли рядом с uvicorn-запросами.
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.analysis.router import router as analysis_router
from app.auth.router import router as auth_router
from app.config import Settings, get_settings
from app.db.alembic_runner import run_alembic_upgrade_to_head
from app.db.session import init_schema
from app.sources.router import router as sources_router

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# Потоки без сессии (нет cookie hpc_csrf) — CSRF-заголовок недоступен, эндпоинты
# должны оставаться доступными при ``csrf_required=True`` (см. docker-прод).
_CSRF_EXEMPT_PATHS = (
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/refresh",
    "/api/auth/verify-email",
    "/api/auth/forgot-password",
    "/api/auth/reset-password",
)


def _csrf_guard(request: Request, settings: Settings) -> JSONResponse | None:
    """Простая проверка CSRF через middleware. Возвращает ответ-отказ
    или ``None`` если всё хорошо."""

    if not settings.csrf_required:
        return None
    if request.method in _SAFE_METHODS:
        return None
    path = request.url.path
    if not path.startswith("/api/"):
        return None
    if any(path.endswith(p.split("/api")[-1]) or path == p for p in _CSRF_EXEMPT_PATHS):
        return None
    cookie = request.cookies.get(settings.csrf_cookie_name)
    header = request.headers.get(settings.csrf_header_name)
    if not cookie or not header or not secrets.compare_digest(cookie, header):
        return JSONResponse(
            status_code=403,
            content={"detail": "CSRF token mismatch"},
        )
    return None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.use_alembic:
            run_alembic_upgrade_to_head(settings)
        else:
            init_schema(settings)
        settings.storage_root.mkdir(parents=True, exist_ok=True)
        yield

    app = FastAPI(
        title=settings.app_name,
        version="0.3.0",
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

    @app.middleware("http")
    async def _csrf_middleware(request: Request, call_next):
        # Читаем settings динамически: тесты могут менять `csrf_required`
        # уже после создания приложения.
        response = _csrf_guard(request, get_settings())
        if response is not None:
            return response
        return await call_next(request)

    @app.get("/api/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok", "app": settings.app_name}

    app.include_router(auth_router, prefix="/api")
    app.include_router(sources_router, prefix="/api")
    app.include_router(analysis_router, prefix="/api")

    return app


app = create_app()
