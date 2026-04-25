"""Бизнес-операции auth, отделённые от HTTP-слоя."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.codes import (
    expires_at,
    generate_numeric_code,
    hash_code,
    verify_code,
)
from app.auth.security import hash_password, verify_password
from app.config import Settings
from app.db.models import User


class AuthError(Exception):
    """Доменная ошибка auth (неверные данные, занятый email и т.п.)."""


def _now() -> datetime:
    return datetime.now(UTC)


def _ensure_aware(dt: datetime | None) -> datetime | None:
    """SQLite не хранит часовой пояс, поэтому при чтении мы получаем naive
    datetime. Приводим к UTC, чтобы сравнения работали и в SQLite, и в
    Postgres."""

    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def get_user_by_email(db: Session, email: str) -> User | None:
    stmt = select(User).where(User.email == email.lower())
    return db.execute(stmt).scalar_one_or_none()


def register_user(
    db: Session,
    email: str,
    password: str,
    *,
    accept_terms: bool,
    accept_pdn: bool,
) -> User:
    if not (accept_terms and accept_pdn):
        # Pydantic уже отрезает такие запросы; держим guard и здесь, чтобы
        # никакой код не мог обойти LEGAL-REG-1.
        raise AuthError("Не отмечены обязательные согласия.")

    existing = get_user_by_email(db, email)
    if existing:
        raise AuthError("Пользователь с таким email уже существует.")
    now = _now()
    user = User(
        email=email.lower(),
        password_hash=hash_password(password),
        terms_accepted_at=now,
        pdn_accepted_at=now,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate(db: Session, email: str, password: str) -> User:
    user = get_user_by_email(db, email)
    if user is None or not verify_password(password, user.password_hash):
        raise AuthError("Неверный email или пароль.")
    return user


# ---------------------------------------------------------------------------
# Email verification (TASK_SPEC_010)
# ---------------------------------------------------------------------------


def issue_email_verification_code(
    db: Session, user: User, settings: Settings
) -> str:
    """Сгенерировать новый код подтверждения email.

    Старый код становится недействителен сразу при перевыдаче.
    Возвращает plaintext код — вызывающая сторона отправляет его через
    mail-sink, в БД остаётся только hash.
    """

    if user.email_verified_at is not None:
        raise AuthError("Email уже подтверждён.")

    code = generate_numeric_code(settings.email_code_length)
    user.email_verification_code_hash = hash_code(code, settings)
    user.email_verification_code_expires_at = expires_at(
        settings.email_code_ttl_minutes
    )
    user.email_verification_code_sent_at = _now()
    db.commit()
    return code


def confirm_email_verification(
    db: Session, user: User, code: str, settings: Settings
) -> User:
    if user.email_verified_at is not None:
        return user
    expires = _ensure_aware(user.email_verification_code_expires_at)
    if (
        user.email_verification_code_hash is None
        or expires is None
        or expires < _now()
    ):
        raise AuthError("Код подтверждения недействителен или истёк.")
    if not verify_code(
        code.strip(), user.email_verification_code_hash, settings
    ):
        raise AuthError("Неверный код подтверждения.")

    user.email_verified_at = _now()
    user.email_verification_code_hash = None
    user.email_verification_code_expires_at = None
    db.commit()
    return user


# ---------------------------------------------------------------------------
# Password reset (TASK_SPEC_010)
# ---------------------------------------------------------------------------


def issue_password_reset_code(
    db: Session, user: User, settings: Settings
) -> str:
    code = generate_numeric_code(settings.email_code_length)
    user.password_reset_code_hash = hash_code(code, settings)
    user.password_reset_code_expires_at = expires_at(
        settings.password_reset_code_ttl_minutes
    )
    db.commit()
    return code


def reset_password_with_code(
    db: Session,
    email: str,
    code: str,
    new_password: str,
    settings: Settings,
) -> User:
    user = get_user_by_email(db, email)
    if user is None:
        raise AuthError("Код восстановления недействителен или истёк.")
    expires = _ensure_aware(user.password_reset_code_expires_at)
    if (
        user.password_reset_code_hash is None
        or expires is None
        or expires < _now()
    ):
        raise AuthError("Код восстановления недействителен или истёк.")
    if not verify_code(code.strip(), user.password_reset_code_hash, settings):
        raise AuthError("Код восстановления недействителен или истёк.")

    user.password_hash = hash_password(new_password)
    user.password_reset_code_hash = None
    user.password_reset_code_expires_at = None
    db.commit()
    return user


# ---------------------------------------------------------------------------
# Onboarding / account lifecycle (TASK_SPEC_010)
# ---------------------------------------------------------------------------


def mark_onboarding_complete(db: Session, user: User) -> User:
    if user.onboarding_completed_at is None:
        user.onboarding_completed_at = _now()
        db.commit()
    return user


def delete_account(db: Session, user: User) -> None:
    db.delete(user)
    db.commit()
