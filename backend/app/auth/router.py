"""HTTP-слой auth. Здесь только сериализация, cookie, коды ответа."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.auth import service
from app.auth.deps import current_user
from app.auth.schemas import LoginRequest, RegisterRequest, UserPublic
from app.auth.security import create_access_token
from app.config import Settings, get_settings
from app.db.models import User
from app.db.session import get_db

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


@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    response: Response,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UserPublic:
    try:
        user = service.register_user(db, payload.email, payload.password)
    except service.AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    token = create_access_token(user.id, settings)
    _set_session_cookie(response, token, settings)
    return UserPublic.model_validate(user)


@router.post("/login", response_model=UserPublic)
def login(
    payload: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UserPublic:
    try:
        user = service.authenticate(db, payload.email, payload.password)
    except service.AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc

    token = create_access_token(user.id, settings)
    _set_session_cookie(response, token, settings)
    return UserPublic.model_validate(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    settings: Settings = Depends(get_settings),
) -> Response:
    resp = Response(status_code=status.HTTP_204_NO_CONTENT)
    resp.delete_cookie(key=settings.cookie_name, path="/")
    return resp


@router.get("/me", response_model=UserPublic)
def me(user: User = Depends(current_user)) -> UserPublic:
    return UserPublic.model_validate(user)
