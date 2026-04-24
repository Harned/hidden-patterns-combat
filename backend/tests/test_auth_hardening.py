"""Тесты TASK_SPEC_009: refresh, email verification, rate-limit backend."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings


def _register(client: TestClient, email: str) -> dict:
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "password": "supersecret123"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Refresh-токен
# ---------------------------------------------------------------------------


def test_refresh_issues_new_session(client: TestClient, tmp_settings: Settings) -> None:
    _register(client, "refresh@example.com")
    old_access = client.cookies.get("hpc_session")
    assert old_access

    resp = client.post("/api/auth/refresh")
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "refresh@example.com"
    assert body["csrf_token"]
    # После refresh access-cookie обновилась (может быть той же
    # строкой при одинаковом iat, но должна присутствовать).
    assert client.cookies.get("hpc_session")


def test_refresh_without_cookie_returns_401(client: TestClient) -> None:
    client.cookies.clear()
    resp = client.post("/api/auth/refresh")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------


def test_request_verification_and_verify_email(
    client: TestClient, tmp_settings: Settings
) -> None:
    _register(client, "verify@example.com")
    me = client.get("/api/auth/me").json()
    assert me["email_verified_at"] is None

    # Извлекаем токен напрямую через security-helper (SMTP не настроен).
    from app.auth.security import create_email_verification_token

    token = create_email_verification_token(me["id"], tmp_settings)
    resp = client.get("/api/auth/verify-email", params={"token": token})
    assert resp.status_code == 200
    assert resp.json()["email_verified_at"] is not None

    me_after = client.get("/api/auth/me").json()
    assert me_after["email_verified_at"] is not None


def test_verify_email_rejects_bad_token(client: TestClient) -> None:
    _register(client, "badtoken@example.com")
    resp = client.get("/api/auth/verify-email", params={"token": "not-a-jwt"})
    assert resp.status_code == 400


def test_require_email_verified_blocks_login(
    client: TestClient, tmp_settings: Settings
) -> None:
    _register(client, "strict@example.com")
    tmp_settings.require_email_verified = True
    # Логаутим, чтобы login пошёл заново и попал в проверку.
    client.post("/api/auth/logout")
    client.cookies.clear()
    resp = client.post(
        "/api/auth/login",
        json={"email": "strict@example.com", "password": "supersecret123"},
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Rate-limit backend: in-memory fallback when redis missing
# ---------------------------------------------------------------------------


def test_rate_limit_falls_back_to_memory_without_redis(
    client: TestClient, tmp_settings: Settings
) -> None:
    tmp_settings.rate_limit_enabled = True
    tmp_settings.rate_limit_auth_per_minute = 2
    tmp_settings.rate_limit_backend = "redis"
    tmp_settings.redis_url = "redis://127.0.0.1:59999/0"  # заведомо недоступен
    from app import rate_limit

    rate_limit._build_auth_limiter.cache_clear()  # type: ignore[attr-defined]

    for _ in range(2):
        client.post(
            "/api/auth/login",
            json={"email": "rl2@example.com", "password": "wrong-password"},
        )
    resp = client.post(
        "/api/auth/login",
        json={"email": "rl2@example.com", "password": "wrong-password"},
    )
    assert resp.status_code == 429
