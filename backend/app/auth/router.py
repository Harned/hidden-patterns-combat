"""HTTP-слой auth. Здесь только сериализация, cookie, коды ответа."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from jwt import InvalidTokenError
from sqlalchemy.orm import Session

from app.auth import service
from app.auth.deps import current_user
from app.auth.schemas import LoginRequest, RegisterRequest, UserPublic
from app.auth.security import (
    create_access_token,
    create_email_verification_token,
    create_refresh_token,
    decode_email_verification_token,
    decode_refresh_token,
)
from app.config import Settings, get_settings
from app.db.models import User
from app.db.session import get_db
from app.rate_limit import get_auth_rate_limiter
from app.security import (
    RateLimiter,
    client_ip,
    generate_csrf_token,
    set_csrf_cookie,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


def _set_session_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=settings.access_token_expires_minutes * 60,
        path="/",
    )


def _set_refresh_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=settings.refresh_token_expires_minutes * 60,
        path="/api/auth",
    )


def _enforce_rate_limit(
    request: Request,
    settings: Settings,
    limiter: RateLimiter,
    bucket_key: str,
) -> None:
    if not settings.rate_limit_enabled:
        return
    ip = client_ip(request)
    if not limiter.check(ip, bucket_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Слишком много попыток. Попробуйте позже.",
        )


def _user_to_public(user: User, csrf: str | None = None) -> UserPublic:
    payload = UserPublic.model_validate(user).model_dump()
    if csrf is not None:
        payload["csrf_token"] = csrf
    return UserPublic.model_validate(payload)


def _issue_session(
    response: Response,
    settings: Settings,
    user: User,
) -> UserPublic:
    access = create_access_token(user.id, settings)
    refresh = create_refresh_token(user.id, settings)
    _set_session_cookie(response, access, settings)
    _set_refresh_cookie(response, refresh, settings)
    csrf = generate_csrf_token()
    set_csrf_cookie(response, csrf, settings)
    return _user_to_public(user, csrf)


@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
def register(
    request: Request,
    response: Response,
    payload: RegisterRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    limiter: RateLimiter = Depends(get_auth_rate_limiter),
) -> UserPublic:
    _enforce_rate_limit(request, settings, limiter, "register")
    try:
        user = service.register_user(db, payload.email, payload.password)
    except service.AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    _issue_verification_email(user, settings)
    return _issue_session(response, settings, user)


@router.post("/login", response_model=UserPublic)
def login(
    request: Request,
    response: Response,
    payload: LoginRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    limiter: RateLimiter = Depends(get_auth_rate_limiter),
) -> UserPublic:
    _enforce_rate_limit(request, settings, limiter, "login")
    try:
        user = service.authenticate(db, payload.email, payload.password)
    except service.AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc

    if settings.require_email_verified and user.email_verified_at is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email не подтверждён. Проверьте почту.",
        )
    return _issue_session(response, settings, user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    settings: Settings = Depends(get_settings),
) -> Response:
    resp = Response(status_code=status.HTTP_204_NO_CONTENT)
    resp.delete_cookie(key=settings.cookie_name, path="/")
    resp.delete_cookie(key=settings.csrf_cookie_name, path="/")
    resp.delete_cookie(key=settings.refresh_cookie_name, path="/api/auth")
    return resp


@router.get("/me", response_model=UserPublic)
def me(user: User = Depends(current_user)) -> UserPublic:
    return _user_to_public(user)


# ---------------------------------------------------------------------------
# Refresh token (TASK_SPEC_009)
# ---------------------------------------------------------------------------


@router.post("/refresh", response_model=UserPublic)
def refresh(
    response: Response,
    hpc_refresh: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UserPublic:
    if not hpc_refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Нет refresh-токена."
        )
    try:
        payload = decode_refresh_token(hpc_refresh, settings)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh-токен недействителен или истёк.",
        ) from exc

    try:
        user_id = int(payload.get("sub", ""))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Некорректный refresh-токен."
        ) from exc

    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Пользователь не найден."
        )

    return _issue_session(response, settings, user)


# ---------------------------------------------------------------------------
# Email verification (TASK_SPEC_009)
# ---------------------------------------------------------------------------


def _issue_verification_email(user: User, settings: Settings) -> None:
    """Заглушка отправки email: логируем ссылку в stdout.

    Реальная SMTP-отправка — вне MVP, добавим в отдельном таске.
    """

    token = create_email_verification_token(user.id, settings)
    logger.info(
        "[email-stub] verification_link user=%s token=%s",
        user.email,
        token,
    )


@router.post("/request-verification", status_code=status.HTTP_202_ACCEPTED)
def request_verification(
    settings: Settings = Depends(get_settings),
    user: User = Depends(current_user),
) -> dict[str, str]:
    if user.email_verified_at is not None:
        return {"status": "already_verified"}
    _issue_verification_email(user, settings)
    return {"status": "sent"}


@router.get("/verify-email", response_model=UserPublic)
def verify_email(
    token: str,
    response: Response,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UserPublic:
    try:
        payload = decode_email_verification_token(token, settings)
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Неверный или просроченный токен подтверждения.",
        ) from exc
    try:
        user_id = int(payload.get("sub", ""))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Некорректный токен подтверждения.",
        ) from exc

    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Пользователь не найден."
        )
    if user.email_verified_at is None:
        user.email_verified_at = datetime.now(UTC)
        db.commit()
    return _issue_session(response, settings, user)
