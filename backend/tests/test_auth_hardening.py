"""Тесты TASK_SPEC_009/010: refresh, email verification (codes), rate-limit backend."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings

_BASE = {
    "password": "supersecret123",
    "accept_terms": True,
    "accept_pdn": True,
}


def _register(client: TestClient, email: str) -> dict:
    resp = client.post("/api/auth/register", json={"email": email, **_BASE})
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Refresh-токен
# ---------------------------------------------------------------------------


def test_refresh_issues_new_session(client: TestClient, tmp_settings: Settings) -> None:
    _register(client, "refresh@example.com")
    assert client.cookies.get("hpc_session")

    resp = client.post("/api/auth/refresh")
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "refresh@example.com"
    assert body["csrf_token"]
    assert client.cookies.get("hpc_session")


def test_refresh_without_cookie_returns_401(client: TestClient) -> None:
    client.cookies.clear()
    resp = client.post("/api/auth/refresh")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Email verification (TASK_SPEC_010, code-based)
# ---------------------------------------------------------------------------


def test_register_creates_unverified_user(
    client: TestClient, tmp_settings: Settings
) -> None:
    body = _register(client, "verify@example.com")
    assert body["email_verified_at"] is None


def test_verify_email_with_code(client: TestClient, tmp_settings: Settings) -> None:
    """Регистрация → выдача кода → POST /verify-email с правильным кодом."""

    from app.auth import service as auth_service
    from app.db.models import User
    from app.db.session import get_sessionmaker

    _register(client, "code@example.com")
    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        user = (
            db.query(User).filter(User.email == "code@example.com").one()
        )
        # Перевыдаём код, чтобы получить plaintext.
        # Для предыдущего кода у нас только hash.
        new_code = auth_service.issue_email_verification_code(db, user, tmp_settings)

    resp = client.post("/api/auth/verify-email", json={"code": new_code})
    assert resp.status_code == 200, resp.text
    assert resp.json()["email_verified_at"] is not None


def test_verify_email_wrong_code(client: TestClient, tmp_settings: Settings) -> None:
    _register(client, "wrongcode@example.com")
    resp = client.post("/api/auth/verify-email", json={"code": "000000"})
    assert resp.status_code == 400


def test_resend_verification(client: TestClient, tmp_settings: Settings) -> None:
    _register(client, "resend@example.com")
    resp = client.post("/api/auth/resend-verification")
    assert resp.status_code == 202


def test_resend_for_already_verified_is_idempotent(
    client: TestClient, tmp_settings: Settings, register_verified
) -> None:
    register_verified("already@example.com")
    resp = client.post("/api/auth/resend-verification")
    assert resp.status_code == 202
    assert resp.json()["status"] == "already_verified"


# ---------------------------------------------------------------------------
# Password reset (TASK_SPEC_010)
# ---------------------------------------------------------------------------


def test_forgot_password_neutral_for_unknown_email(
    client: TestClient, tmp_settings: Settings
) -> None:
    """Ответ должен быть нейтральным независимо от существования аккаунта."""

    resp = client.post(
        "/api/auth/forgot-password", json={"email": "no-such@example.com"}
    )
    assert resp.status_code == 202
    assert "если" in resp.json()["message"].lower()


def test_password_reset_full_flow(
    client: TestClient, tmp_settings: Settings, register_verified
) -> None:
    """Запрос кода → вытащить из БД → reset → login с новым паролем."""

    from app.auth import service as auth_service
    from app.db.models import User
    from app.db.session import get_sessionmaker

    register_verified("pw@example.com")
    client.cookies.clear()

    # Запрос кода (нейтральный ответ).
    client.post("/api/auth/forgot-password", json={"email": "pw@example.com"})
    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == "pw@example.com").one()
        # Перевыдаём код, чтобы вытащить plaintext.
        code = auth_service.issue_password_reset_code(db, user, tmp_settings)

    resp = client.post(
        "/api/auth/reset-password",
        json={
            "email": "pw@example.com",
            "code": code,
            "new_password": "brand-new-pw-1234",
            "new_password_repeat": "brand-new-pw-1234",
        },
    )
    assert resp.status_code == 200

    # Старый пароль больше не подходит.
    bad = client.post(
        "/api/auth/login",
        json={"email": "pw@example.com", "password": "supersecret123"},
    )
    assert bad.status_code == 401

    # Новый пароль работает.
    good = client.post(
        "/api/auth/login",
        json={"email": "pw@example.com", "password": "brand-new-pw-1234"},
    )
    assert good.status_code == 200


def test_password_reset_mismatched_passwords(
    client: TestClient, tmp_settings: Settings, register_verified
) -> None:
    register_verified("mismatch@example.com")
    client.cookies.clear()
    resp = client.post(
        "/api/auth/reset-password",
        json={
            "email": "mismatch@example.com",
            "code": "123456",
            "new_password": "aaaaaaaa11",
            "new_password_repeat": "bbbbbbbb22",
        },
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------------------


def test_onboarding_complete(
    client: TestClient, tmp_settings: Settings, register_verified
) -> None:
    register_verified("onb@example.com")
    me = client.get("/api/auth/me").json()
    assert me["onboarding_completed_at"] is None

    resp = client.post("/api/auth/onboarding-complete")
    assert resp.status_code == 200
    assert resp.json()["onboarding_completed_at"] is not None


# ---------------------------------------------------------------------------
# Account deletion
# ---------------------------------------------------------------------------


def test_delete_account_removes_user_and_session(
    client: TestClient, tmp_settings: Settings, register_verified
) -> None:
    register_verified("kill@example.com")
    resp = client.delete("/api/auth/me")
    assert resp.status_code == 204
    client.cookies.clear()
    me = client.get("/api/auth/me")
    assert me.status_code == 401


# ---------------------------------------------------------------------------
# Rate-limit backend: in-memory fallback when redis missing
# ---------------------------------------------------------------------------


def test_rate_limit_falls_back_to_memory_without_redis(
    client: TestClient, tmp_settings: Settings
) -> None:
    tmp_settings.rate_limit_enabled = True
    tmp_settings.rate_limit_auth_per_minute = 2
    tmp_settings.rate_limit_backend = "redis"
    tmp_settings.redis_url = "redis://127.0.0.1:59999/0"
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
