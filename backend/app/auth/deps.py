"""FastAPI-зависимости auth: чтение cookie, выдача текущего пользователя."""

from __future__ import annotations

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.security import decode_access_token
from app.config import Settings, get_settings
from app.db.models import User
from app.db.session import get_db


def _extract_token(
    cookie_name: str,
    session_cookie: str | None,
) -> str | None:
    _ = cookie_name  # сохраним сигнатуру для future-расширений
    return session_cookie


def current_user(
    hpc_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> User:
    """Обязательная зависимость: возвращает пользователя или 401."""

    token = _extract_token(settings.cookie_name, hpc_session)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Требуется авторизация.",
        )
    try:
        payload = decode_access_token(token, settings)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Сессия недействительна или истекла.",
        ) from exc

    user_id_raw = payload.get("sub")
    if not user_id_raw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Сессия недействительна.",
        )
    try:
        user_id = int(user_id_raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Сессия недействительна.",
        ) from exc

    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден.",
        )
    return user
