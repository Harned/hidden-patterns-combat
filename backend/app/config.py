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

    secret_key: str = Field(
        default="dev-only-change-me",
        description="Секрет для подписи JWT. В проде задать через HPC_SECRET_KEY.",
    )
    access_token_expires_minutes: int = 60 * 24  # 24 часа
    cookie_name: str = "hpc_session"
    cookie_secure: bool = False  # включить True в проде под HTTPS
    cookie_samesite: str = "lax"

    storage_root: Path = PROJECT_ROOT / "storage" / "uploads"
    max_upload_size_bytes: int = 50 * 1024 * 1024  # 50 MiB

    allowed_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def get_settings() -> Settings:
    """Функция-фабрика, удобна как Depends и для переопределения в тестах."""

    return Settings()  # type: ignore[call-arg]
