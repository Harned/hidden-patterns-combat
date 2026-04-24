"""CSRF и rate-limit — лёгкие in-process реализации (TASK_SPEC_006).

Оба механизма намеренно простые (in-memory), чтобы MVP работал без
внешних зависимостей (Redis и т.п.). Замена на распределённые
реализации — отдельная задача.
"""

from __future__ import annotations

import secrets
import time
from collections import defaultdict
from threading import Lock

from fastapi import Cookie, Header, HTTPException, Request, status

from app.config import Settings

# ---------------------------------------------------------------------------
# CSRF
# ---------------------------------------------------------------------------

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def set_csrf_cookie(response, token: str, settings: Settings) -> None:
    """Выставить CSRF-cookie (НЕ HttpOnly — фронту нужно читать из JS)."""

    response.set_cookie(
        key=settings.csrf_cookie_name,
        value=token,
        httponly=False,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=settings.access_token_expires_minutes * 60,
        path="/",
    )


async def enforce_csrf(
    request: Request,
    settings: Settings,
    hpc_csrf: str | None = Cookie(default=None),
    x_csrf_token: str | None = Header(default=None),
) -> None:
    """FastAPI-зависимость: проверяет double-submit CSRF для мутирующих
    запросов. В dev/test (`csrf_required = False`) — no-op."""

    if not settings.csrf_required:
        return
    if request.method in _SAFE_METHODS:
        return
    # `/auth/login` и `/auth/register` не требуют CSRF по факту, потому что
    # cookie ещё не установлена. В остальных случаях — требуем совпадения.
    path = request.url.path
    if path.endswith("/auth/login") or path.endswith("/auth/register"):
        return
    if not hpc_csrf or not x_csrf_token or not secrets.compare_digest(
        hpc_csrf, x_csrf_token
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token mismatch",
        )


# ---------------------------------------------------------------------------
# Rate limit (token bucket per IP per endpoint-group)
# ---------------------------------------------------------------------------


class _Bucket:
    __slots__ = ("hits", "reset_at")

    def __init__(self) -> None:
        self.hits: int = 0
        self.reset_at: float = 0.0


class RateLimiter:
    """Протокол для rate-limiter'а. Реализации ниже (in-memory / Redis)
    предоставляют одинаковый метод :meth:`check`."""

    def check(self, ip: str, bucket_key: str) -> bool:
        raise NotImplementedError


class InMemoryRateLimiter(RateLimiter):
    """In-memory token bucket per (ip, bucket_key) за минуту.

    Не предназначен для многопроцессного backend'а. Для horizontal
    scaling используйте Redis-адаптер.
    """

    def __init__(self, max_hits_per_minute: int) -> None:
        self.max = max_hits_per_minute
        self._lock = Lock()
        self._state: dict[tuple[str, str], _Bucket] = defaultdict(_Bucket)

    def check(self, ip: str, bucket_key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            bucket = self._state[(ip, bucket_key)]
            if now >= bucket.reset_at:
                bucket.hits = 0
                bucket.reset_at = now + 60.0
            if bucket.hits >= self.max:
                return False
            bucket.hits += 1
            return True


class RedisRateLimiter(RateLimiter):
    """Атомарный INCR+EXPIRE per-minute по ключу `rl:{bucket}:{ip}:{minute}`."""

    def __init__(self, redis_url: str, max_hits_per_minute: int) -> None:
        import redis  # локальный импорт: зависимость optional

        self.max = max_hits_per_minute
        self.client = redis.Redis.from_url(
            redis_url, decode_responses=True, socket_connect_timeout=2
        )
        # проверим соединение раньше, чтобы фабрика могла упасть
        # на in-memory fallback.
        self.client.ping()

    def check(self, ip: str, bucket_key: str) -> bool:
        now_minute = int(time.time() // 60)
        key = f"rl:{bucket_key}:{ip}:{now_minute}"
        pipe = self.client.pipeline()
        pipe.incr(key, 1)
        pipe.expire(key, 65)
        hits, _ = pipe.execute()
        return int(hits) <= self.max


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
