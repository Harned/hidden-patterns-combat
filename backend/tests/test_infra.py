"""Тесты инфраструктуры (TASK_SPEC_006): фоновый анализ, CSRF, rate-limit."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings

_REG_PAYLOAD = {
    "password": "supersecret123",
    "accept_terms": True,
    "accept_pdn": True,
}


def _register(client: TestClient, email: str) -> None:
    resp = client.post(
        "/api/auth/register",
        json={"email": email, **_REG_PAYLOAD},
    )
    assert resp.status_code == 201


def test_register_returns_csrf_token(client: TestClient) -> None:
    resp = client.post(
        "/api/auth/register",
        json={"email": "csrf@example.com", **_REG_PAYLOAD},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["csrf_token"] and isinstance(body["csrf_token"], str)
    assert "hpc_csrf" in resp.cookies


def test_csrf_forgot_and_reset_work_without_token_when_csrf_on(
    tmp_settings: Settings,
    client: TestClient,
) -> None:
    """Сброс пароля — без сессии; нельзя требовать X-CSRF-Token (cookie нет)."""

    tmp_settings.csrf_required = True
    _register(client, "forgot-csrf@example.com")
    client.cookies.clear()
    r1 = client.post(
        "/api/auth/forgot-password", json={"email": "forgot-csrf@example.com"}
    )
    assert r1.status_code == 202, r1.text
    r2 = client.post(
        "/api/auth/reset-password",
        json={
            "email": "forgot-csrf@example.com",
            "code": "000000",
            "new_password": "nope-fail-11",
            "new_password_repeat": "nope-fail-11",
        },
    )
    # Код неверный — важен не 403 CSRF, а 400 с доменной ошибкой.
    assert r2.status_code == 400, r2.text
    assert "недейств" in r2.json()["detail"].lower()


def test_csrf_required_blocks_mutations(
    tmp_settings: Settings,
    client: TestClient,
    register_verified,
    multirow_xlsx_bytes: bytes,
) -> None:
    """В prod-режиме (csrf_required=True) мутирующий запрос без
    заголовка X-CSRF-Token должен получать 403."""

    tmp_settings.csrf_required = True
    register_verified("csrf-strict@example.com")
    resp = client.post(
        "/api/sources",
        files={
            "file": (
                "x.xlsx",
                multirow_xlsx_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
        data={"confirm_upload": "true"},
    )
    assert resp.status_code == 403


def test_csrf_passes_with_matching_header(
    tmp_settings: Settings,
    client: TestClient,
    register_verified,
    multirow_xlsx_bytes: bytes,
) -> None:
    tmp_settings.csrf_required = True
    register_verified("csrf-ok@example.com")
    csrf = client.cookies.get("hpc_csrf")
    assert csrf

    resp = client.post(
        "/api/sources",
        headers={"X-CSRF-Token": csrf},
        files={
            "file": (
                "y.xlsx",
                multirow_xlsx_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
        data={"confirm_upload": "true"},
    )
    assert resp.status_code == 201


def test_rate_limit_login_returns_429(
    tmp_settings: Settings, client: TestClient
) -> None:
    tmp_settings.rate_limit_enabled = True
    tmp_settings.rate_limit_auth_per_minute = 3
    from app import rate_limit

    rate_limit._build_auth_limiter.cache_clear()  # type: ignore[attr-defined]

    for _ in range(3):
        client.post(
            "/api/auth/login",
            json={"email": "rl@example.com", "password": "wrong-password"},
        )
    resp = client.post(
        "/api/auth/login",
        json={"email": "rl@example.com", "password": "wrong-password"},
    )
    assert resp.status_code == 429


def test_background_analyze_reaches_done(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    register_verified("bg@example.com")
    up = client.post(
        "/api/sources",
        files={
            "file": (
                "bg.xlsx",
                multirow_xlsx_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
        data={"confirm_upload": "true"},
    )
    sid = up.json()["id"]
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)
    client.post(f"/api/sources/{sid}/finalize")

    resp = client.post(f"/api/sources/{sid}/analyze")
    assert resp.status_code == 200
    run = resp.json()
    run_id = run["id"]
    polled = client.get(f"/api/sources/{sid}/runs/{run_id}").json()
    assert polled["state"] == "done", polled
    assert polled["status"] == "baseline_only"


def test_latest_result_ignores_failed_runs(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    register_verified("noresult@example.com")
    up = client.post(
        "/api/sources",
        files={
            "file": (
                "n.xlsx",
                multirow_xlsx_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
        data={"confirm_upload": "true"},
    )
    sid = up.json()["id"]
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)
    client.post(f"/api/sources/{sid}/finalize")
    resp = client.post(f"/api/sources/{sid}/analyze", params={"wait": "true"})
    assert resp.status_code == 200
    run_id = resp.json()["id"]

    from app.db.models import AnalysisRun
    from app.db.session import get_sessionmaker

    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        run = db.get(AnalysisRun, run_id)
        run.state = "failed"
        run.error = "simulated"
        db.commit()

    resp = client.get(f"/api/sources/{sid}/result")
    assert resp.status_code == 404
