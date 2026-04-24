"""Бизнес-операции auth, отделённые от HTTP-слоя."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.security import hash_password, verify_password
from app.db.models import User


class AuthError(Exception):
    """Доменная ошибка auth (неверные данные, занятый email и т.п.)."""


def get_user_by_email(db: Session, email: str) -> User | None:
    stmt = select(User).where(User.email == email.lower())
    return db.execute(stmt).scalar_one_or_none()


def register_user(db: Session, email: str, password: str) -> User:
    existing = get_user_by_email(db, email)
    if existing:
        raise AuthError("Пользователь с таким email уже существует.")
    user = User(email=email.lower(), password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate(db: Session, email: str, password: str) -> User:
    user = get_user_by_email(db, email)
    if user is None or not verify_password(password, user.password_hash):
        raise AuthError("Неверный email или пароль.")
    return user
