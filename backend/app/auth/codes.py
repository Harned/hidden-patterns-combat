"""Утилиты для коротких email-кодов (verification / password reset).

* Сгенерированный код (например, ``"482917"``) хранится в БД **только**
  в виде HMAC-SHA256 от ``settings.secret_key`` — пользователь видит код
  один раз, в письме.
* Сверка — через :func:`verify_code` за константное время
  (``hmac.compare_digest``).
* Секрет — общий с JWT, что приемлемо для локального MVP. Для прода
  можно вынести отдельный ``HPC_CODE_SECRET``.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta

from app.config import Settings


def generate_numeric_code(length: int = 6) -> str:
    """Сгенерировать криптографически случайный числовой код заданной длины."""

    if length <= 0:
        raise ValueError("length must be positive")
    upper = 10**length
    value = secrets.randbelow(upper)
    return str(value).zfill(length)


def hash_code(code: str, settings: Settings) -> str:
    return hmac.new(
        settings.secret_key.encode("utf-8"),
        code.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_code(code: str, hashed: str | None, settings: Settings) -> bool:
    if not hashed:
        return False
    return hmac.compare_digest(hash_code(code, settings), hashed)


def expires_at(minutes: int) -> datetime:
    return datetime.now(UTC) + timedelta(minutes=minutes)
