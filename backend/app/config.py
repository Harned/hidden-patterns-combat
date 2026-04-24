"""Конфигурация backend (без секретов в коде)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Настройки приложения.

    Значения по умолчанию рассчитаны на локальный dev-режим. Все чувствительные
    параметры должны приходить через переменные окружения (`HPC_*`).
    """

    model_config = SettingsConfigDict(
        env_prefix="HPC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "hidden-patterns-combat backend"
    environment: str = "dev"
    debug: bool = True

    database_url: str = f"sqlite:///{PROJECT_ROOT / 'storage' / 'app.db'}"
    # Если True, backend не создаёт таблицы через create_all и ожидает,
    # что Alembic уже применил миграции. В тестах по умолчанию False.
    use_alembic: bool = False

    secret_key: str = Field(
        default="dev-only-change-me",
        description="Секрет для подписи JWT. В проде задать через HPC_SECRET_KEY.",
    )
    access_token_expires_minutes: int = 15  # короткий access
    refresh_token_expires_minutes: int = 60 * 24 * 14  # 14 дней
    cookie_name: str = "hpc_session"
    refresh_cookie_name: str = "hpc_refresh"
    cookie_secure: bool = False  # включить True в проде под HTTPS
    cookie_samesite: str = "lax"

    # CSRF: double-submit cookie. Отдельное non-HttpOnly cookie +
    # заголовок `X-CSRF-Token` на мутирующих запросах.
    csrf_cookie_name: str = "hpc_csrf"
    csrf_required: bool = False  # в dev/test — False, в production переключать на True
    csrf_header_name: str = "X-CSRF-Token"

    # Rate-limit на auth-эндпоинты.
    rate_limit_enabled: bool = False  # в dev/test — False
    rate_limit_auth_per_minute: int = 5
    rate_limit_backend: str = "memory"  # memory | redis
    redis_url: str | None = None

    # Email verification. Реальная SMTP-отправка не реализована —
    # токены логируются через stdout. См. `TASK_SPEC_009`.
    require_email_verified: bool = False
    email_verification_token_minutes: int = 60 * 24  # сутки

    # Фоновая задача analyze: верхняя граница времени, после которой
    # внешний watcher может пометить run как stuck (не используется
    # в in-process BackgroundTasks, но полезно зафиксировать.)
    analyze_timeout_seconds: int = 600

    storage_root: Path = PROJECT_ROOT / "storage" / "uploads"
    max_upload_size_bytes: int = 50 * 1024 * 1024  # 50 MiB

    allowed_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def get_settings() -> Settings:
    """Функция-фабрика, удобна как Depends и для переопределения в тестах."""

    return Settings()  # type: ignore[call-arg]
