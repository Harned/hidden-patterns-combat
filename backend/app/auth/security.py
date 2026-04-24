"""Хеширование паролей и подпись JWT."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from passlib.context import CryptContext

from app.config import Settings, get_settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

_ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _pwd_context.verify(password, password_hash)


def _encode(
    subject: str | int,
    minutes: int,
    typ: str,
    settings: Settings,
    extra: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "typ": typ,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=minutes)).timestamp()),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.secret_key, algorithm=_ALGORITHM)


def create_access_token(
    subject: str | int,
    settings: Settings | None = None,
    expires_minutes: int | None = None,
) -> str:
    settings = settings or get_settings()
    minutes = (
        expires_minutes
        if expires_minutes is not None
        else settings.access_token_expires_minutes
    )
    return _encode(subject, minutes, "access", settings)


def create_refresh_token(
    subject: str | int,
    settings: Settings | None = None,
) -> str:
    settings = settings or get_settings()
    return _encode(subject, settings.refresh_token_expires_minutes, "refresh", settings)


def create_email_verification_token(
    subject: str | int,
    settings: Settings | None = None,
) -> str:
    settings = settings or get_settings()
    return _encode(
        subject, settings.email_verification_token_minutes, "email_verify", settings
    )


def decode_token(token: str, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    return jwt.decode(token, settings.secret_key, algorithms=[_ALGORITHM])


def decode_access_token(token: str, settings: Settings | None = None) -> dict[str, Any]:
    payload = decode_token(token, settings)
    if payload.get("typ") not in (None, "access"):
        raise jwt.InvalidTokenError(
            f"Expected access token, got typ={payload.get('typ')}"
        )
    return payload


def decode_refresh_token(token: str, settings: Settings | None = None) -> dict[str, Any]:
    payload = decode_token(token, settings)
    if payload.get("typ") != "refresh":
        raise jwt.InvalidTokenError(
            f"Expected refresh token, got typ={payload.get('typ')}"
        )
    return payload


def decode_email_verification_token(
    token: str, settings: Settings | None = None
) -> dict[str, Any]:
    payload = decode_token(token, settings)
    if payload.get("typ") != "email_verify":
        raise jwt.InvalidTokenError(
            f"Expected email verification token, got typ={payload.get('typ')}"
        )
    return payload
