"""HTTP-слой auth. Здесь только сериализация, cookie, коды ответа."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.auth import service
from app.auth.deps import current_user
from app.auth.schemas import (
    ForgotPasswordRequest,
    LoginRequest,
    RegisterRequest,
    ResetPasswordRequest,
    UserPublic,
    VerifyEmailRequest,
)
from app.auth.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
)
from app.config import Settings, get_settings
from app.db.models import User
from app.db.session import get_db
from app.mail import (
    send_email_verification_code,
    send_password_reset_code,
)
from app.rate_limit import get_auth_rate_limiter
from app.security import (
    RateLimiter,
    client_ip,
    generate_csrf_token,
    set_csrf_cookie,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# cookie / csrf / rate-limit helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Registration & login
# ---------------------------------------------------------------------------


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
        user = service.register_user(
            db,
            payload.email,
            payload.password,
            accept_terms=payload.accept_terms,
            accept_pdn=payload.accept_pdn,
        )
    except service.AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    # Сразу выдаём код подтверждения; UI откроет экран ввода кода.
    code = service.issue_email_verification_code(db, user, settings)
    send_email_verification_code(settings, user.email, code)
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

    # AUTH-LOGIN-1: сессия выдаётся всегда; решение о доступе к рабочей
    # области принимается роут-гардами (current_verified_user) и UI'ем
    # на основе `email_verified_at`.
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


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
def delete_me(
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Самостоятельное удаление аккаунта (PROFILE-1).

    Каскадно удаляет источники пользователя (через ondelete=CASCADE).
    Сбрасывает session/refresh/csrf cookies.
    """

    service.delete_account(db, user)
    resp = Response(status_code=status.HTTP_204_NO_CONTENT)
    resp.delete_cookie(key=settings.cookie_name, path="/")
    resp.delete_cookie(key=settings.csrf_cookie_name, path="/")
    resp.delete_cookie(key=settings.refresh_cookie_name, path="/api/auth")
    return resp


# ---------------------------------------------------------------------------
# Refresh token
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
# Email verification (TASK_SPEC_010)
# ---------------------------------------------------------------------------


@router.post("/verify-email", response_model=UserPublic)
def verify_email(
    payload: VerifyEmailRequest,
    response: Response,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> UserPublic:
    try:
        service.confirm_email_verification(db, user, payload.code, settings)
    except service.AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    return _issue_session(response, settings, user)


@router.post("/resend-verification", status_code=status.HTTP_202_ACCEPTED)
def resend_verification(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    settings: Settings = Depends(get_settings),
    limiter: RateLimiter = Depends(get_auth_rate_limiter),
) -> dict[str, str]:
    if user.email_verified_at is not None:
        return {"status": "already_verified"}
    _enforce_rate_limit(request, settings, limiter, "resend_verification")
    try:
        code = service.issue_email_verification_code(db, user, settings)
    except service.AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    send_email_verification_code(settings, user.email, code)
    return {"status": "sent"}


# ---------------------------------------------------------------------------
# Password reset (TASK_SPEC_010)
# ---------------------------------------------------------------------------


@router.post("/forgot-password", status_code=status.HTTP_202_ACCEPTED)
def forgot_password(
    request: Request,
    payload: ForgotPasswordRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    limiter: RateLimiter = Depends(get_auth_rate_limiter),
) -> dict[str, str]:
    """Запросить код восстановления.

    Ответ всегда **нейтральный**, даже если email не зарегистрирован.
    Это требование AUTH-PWRESET-1 — не утечь, существует ли учётная запись.
    """

    _enforce_rate_limit(request, settings, limiter, "forgot_password")
    user = service.get_user_by_email(db, payload.email)
    if user is not None:
        code = service.issue_password_reset_code(db, user, settings)
        send_password_reset_code(settings, user.email, code)
    else:
        # Не утечь факт существования адреса; в лог — чтобы в dev было видно,
        # почему «код не пришёл» (код в этом случае намеренно не генерируется).
        logger.warning(
            "forgot-password: записи с таким email нет — код в лог не пишем "
            "(проверьте написание адреса)"
        )
    return {
        "status": "ok",
        "message": (
            "Если аккаунт с таким email существует, мы отправили код "
            "восстановления."
        ),
    }


@router.post("/reset-password", status_code=status.HTTP_200_OK)
def reset_password(
    request: Request,
    payload: ResetPasswordRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    limiter: RateLimiter = Depends(get_auth_rate_limiter),
) -> dict[str, str]:
    if payload.new_password != payload.new_password_repeat:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пароли не совпадают.",
        )
    _enforce_rate_limit(request, settings, limiter, "reset_password")
    try:
        service.reset_password_with_code(
            db, payload.email, payload.code, payload.new_password, settings
        )
    except service.AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Onboarding (LEGAL-ONBOARD-1)
# ---------------------------------------------------------------------------


@router.post("/onboarding-complete", response_model=UserPublic)
def onboarding_complete(
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> UserPublic:
    """Отметить, что пользователь увидел и принял первичный дисклеймер.

    Не привязано к согласиям из регистрации — это отдельный шаг
    (LEGAL-ONBOARD-1) о тестовой природе сервиса.
    """

    service.mark_onboarding_complete(db, user)
    return _user_to_public(user)
