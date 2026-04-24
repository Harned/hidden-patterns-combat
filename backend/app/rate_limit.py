"""Глобальный rate-limiter и его DI-фабрика.

Выбор backend'а через ``HPC_RATE_LIMIT_BACKEND``:

* ``memory`` (по умолчанию) — in-process token bucket.
* ``redis`` — атомарный ``INCR`` в Redis; требуется ``HPC_REDIS_URL``
  и пакет ``redis`` (устанавливается автоматически при
  ``pip install -e backend[redis]``).

При недоступности Redis фабрика логирует ошибку и падает к in-memory,
чтобы backend оставался живым — в проде такое событие должно
подниматься алертом.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends

from app.config import Settings, get_settings
from app.security import InMemoryRateLimiter, RateLimiter, RedisRateLimiter

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _build_auth_limiter(
    backend: str,
    max_per_minute: int,
    redis_url: str | None,
) -> RateLimiter:
    if backend == "redis" and redis_url:
        try:
            return RedisRateLimiter(redis_url, max_per_minute)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to initialise Redis rate-limiter (%s); falling back to in-memory",
                exc,
            )
    return InMemoryRateLimiter(max_per_minute)


def get_auth_rate_limiter(
    settings: Settings = Depends(get_settings),
) -> RateLimiter:
    return _build_auth_limiter(
        settings.rate_limit_backend,
        settings.rate_limit_auth_per_minute,
        settings.redis_url,
    )
