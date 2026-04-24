"""SQLAlchemy session management."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import Settings, get_settings


class Base(DeclarativeBase):
    """Общий базовый класс ORM-моделей."""


@lru_cache(maxsize=1)
def _build_engine(database_url: str):
    connect_args: dict[str, object] = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(database_url, future=True, connect_args=connect_args)


def get_engine(settings: Settings | None = None):
    settings = settings or get_settings()
    return _build_engine(settings.database_url)


def get_sessionmaker(settings: Settings | None = None) -> sessionmaker[Session]:
    engine = get_engine(settings)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_schema(settings: Settings | None = None) -> None:
    """Создать таблицы. Для MVP используется вместо Alembic."""

    from app.db import models  # noqa: F401 — регистрация моделей

    engine = get_engine(settings)
    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    """FastAPI-зависимость для получения сессии БД."""

    SessionLocal = get_sessionmaker()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
