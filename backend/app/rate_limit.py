"""Глобальный in-process rate-limiter и его DI-фабрика."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from app.config import Settings, get_settings
from app.security import RateLimiter


@lru_cache(maxsize=1)
def _build_auth_limiter(max_per_minute: int) -> RateLimiter:
    return RateLimiter(max_hits_per_minute=max_per_minute)


def get_auth_rate_limiter(
    settings: Settings = Depends(get_settings),
) -> RateLimiter:
    return _build_auth_limiter(settings.rate_limit_auth_per_minute)
