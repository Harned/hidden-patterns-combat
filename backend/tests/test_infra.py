"""Тесты инфраструктуры (TASK_SPEC_006): фоновый анализ, CSRF, rate-limit."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings


def _register(client: TestClient, email: str) -> None:
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "password": "supersecret123"},
    )
    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# CSRF token выдаётся и уважается
# ---------------------------------------------------------------------------


def test_register_returns_csrf_token(client: TestClient) -> None:
    resp = client.post(
        "/api/auth/register",
        json={"email": "csrf@example.com", "password": "supersecret123"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["csrf_token"] and isinstance(body["csrf_token"], str)
    assert "hpc_csrf" in resp.cookies


def test_csrf_required_blocks_mutations(
    tmp_settings: Settings, client: TestClient, multirow_xlsx_bytes: bytes
) -> None:
    """В prod-режиме (csrf_required=True) мутирующий запрос без
    заголовка X-CSRF-Token должен получать 403."""

    tmp_settings.csrf_required = True
    _register(client, "csrf-strict@example.com")
    # Cookie с CSRF есть, заголовка — нет.
    resp = client.post(
        "/api/sources",
        files={
            "file": (
                "x.xlsx",
                multirow_xlsx_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
    )
    assert resp.status_code == 403


def test_csrf_passes_with_matching_header(
    tmp_settings: Settings, client: TestClient, multirow_xlsx_bytes: bytes
) -> None:
    tmp_settings.csrf_required = True
    register_resp = client.post(
        "/api/auth/register",
        json={"email": "csrf-ok@example.com", "password": "supersecret123"},
    )
    csrf = register_resp.json()["csrf_token"]

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
    )
    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Rate-limit на auth endpoints
# ---------------------------------------------------------------------------


def test_rate_limit_login_returns_429(
    tmp_settings: Settings, client: TestClient
) -> None:
    tmp_settings.rate_limit_enabled = True
    tmp_settings.rate_limit_auth_per_minute = 3
    # Сбросим глобальный limiter-кэш, чтобы тест был изолирован.
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


# ---------------------------------------------------------------------------
# Фоновый analyze через BackgroundTasks
# ---------------------------------------------------------------------------


def test_background_analyze_reaches_done(
    client: TestClient, multirow_xlsx_bytes: bytes
) -> None:
    client.post(
        "/api/auth/register",
        json={"email": "bg@example.com", "password": "supersecret123"},
    )
    up = client.post(
        "/api/sources",
        files={
            "file": (
                "bg.xlsx",
                multirow_xlsx_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
    )
    sid = up.json()["id"]
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)

    resp = client.post(f"/api/sources/{sid}/analyze")
    assert resp.status_code == 200
    run = resp.json()
    run_id = run["id"]
    # В TestClient BackgroundTasks запускаются синхронно после ответа,
    # поэтому последующий GET уже должен видеть done.
    polled = client.get(f"/api/sources/{sid}/runs/{run_id}").json()
    assert polled["state"] == "done", polled
    assert polled["status"] == "baseline_only"


def test_latest_result_ignores_failed_runs(
    client: TestClient, multirow_xlsx_bytes: bytes
) -> None:
    client.post(
        "/api/auth/register",
        json={"email": "noresult@example.com", "password": "supersecret123"},
    )
    up = client.post(
        "/api/sources",
        files={
            "file": (
                "n.xlsx",
                multirow_xlsx_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
    )
    sid = up.json()["id"]
    # Нет mapping — analyze запускается, результат state=done + baseline_only
    # Для проверки «нет result при state != done» прямо в БД выставим state=failed.
    resp = client.post(f"/api/sources/{sid}/analyze", params={"wait": "true"})
    assert resp.status_code == 200
    run_id = resp.json()["id"]

    # Помечаем единственный run как failed, чтобы latest_result вернул 404.
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
