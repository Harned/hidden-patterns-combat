"""HTTP-слой auth. Здесь только сериализация, cookie, коды ответа."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.auth import service
from app.auth.deps import current_user
from app.auth.schemas import LoginRequest, RegisterRequest, UserPublic
from app.auth.security import create_access_token
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


def _issue_session(
    response: Response,
    settings: Settings,
    user: User,
) -> UserPublic:
    token = create_access_token(user.id, settings)
    _set_session_cookie(response, token, settings)
    csrf = generate_csrf_token()
    set_csrf_cookie(response, csrf, settings)
    payload = UserPublic.model_validate(user)
    # CSRF token возвращается в теле, чтобы SPA могла сразу положить
    # его в заголовок следующего мутирующего запроса.
    payload_dict = payload.model_dump()
    payload_dict["csrf_token"] = csrf
    return UserPublic.model_validate(payload_dict)


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

    return _issue_session(response, settings, user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    settings: Settings = Depends(get_settings),
) -> Response:
    resp = Response(status_code=status.HTTP_204_NO_CONTENT)
    resp.delete_cookie(key=settings.cookie_name, path="/")
    resp.delete_cookie(key=settings.csrf_cookie_name, path="/")
    return resp


@router.get("/me", response_model=UserPublic)
def me(user: User = Depends(current_user)) -> UserPublic:
    return UserPublic.model_validate(user)
