"""Тесты TASK_SPEC_007: история запусков и preview листа."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _setup(client: TestClient, register_verified, email: str, data: bytes) -> int:
    register_verified(email)
    resp = client.post(
        "/api/sources",
        files={
            "file": (
                "x.xlsx",
                data,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
        data={"confirm_upload": "true"},
    )
    assert resp.status_code == 201
    return resp.json()["id"]


def test_runs_history_lists_all_runs(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _setup(client, register_verified, "history@example.com", multirow_xlsx_bytes)
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)
    client.post(f"/api/sources/{sid}/finalize")

    run1 = client.post(
        f"/api/sources/{sid}/analyze", params={"wait": "true", "mode": "basic"}
    ).json()
    run2 = client.post(
        f"/api/sources/{sid}/analyze", params={"wait": "true", "mode": "basic"}
    ).json()

    runs = client.get(f"/api/sources/{sid}/runs").json()
    assert isinstance(runs, list)
    assert len(runs) >= 2
    ids = {r["id"] for r in runs}
    assert run1["id"] in ids and run2["id"] in ids
    assert runs[0]["id"] >= runs[-1]["id"]


def test_run_result_endpoint_returns_done_run(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _setup(client, register_verified, "runres@example.com", multirow_xlsx_bytes)
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)
    client.post(f"/api/sources/{sid}/finalize")
    run = client.post(
        f"/api/sources/{sid}/analyze", params={"wait": "true"}
    ).json()
    assert run["state"] == "done"

    full = client.get(f"/api/sources/{sid}/runs/{run['id']}/result").json()
    assert full["id"] == run["id"]
    assert full["result"]["status"] == run["status"]


def test_run_result_endpoint_rejects_non_done(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _setup(client, register_verified, "runconflict@example.com", multirow_xlsx_bytes)
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)
    client.post(f"/api/sources/{sid}/finalize")
    run = client.post(
        f"/api/sources/{sid}/analyze", params={"wait": "true"}
    ).json()
    run_id = run["id"]

    from app.db.models import AnalysisRun
    from app.db.session import get_sessionmaker

    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        obj = db.get(AnalysisRun, run_id)
        obj.state = "failed"
        obj.error = "simulated"
        db.commit()

    resp = client.get(f"/api/sources/{sid}/runs/{run_id}/result")
    assert resp.status_code == 409


def test_sheet_preview_returns_rows_and_columns(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _setup(client, register_verified, "preview@example.com", multirow_xlsx_bytes)
    resp = client.get(
        f"/api/sources/{sid}/sheets/48/preview",
        params={"header_rows": "0,1,2", "rows": "3"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sheet"] == "48"
    assert body["header_rows"] == [0, 1, 2]
    assert "Баллы | ЗАП" in body["columns"]
    assert len(body["preview"]) <= 3
    for row in body["preview"]:
        for value in row.values():
            assert value is None or isinstance(value, (str, int, float, bool))


def test_sheet_preview_enforces_ownership(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _setup(client, register_verified, "preview-owner@example.com", multirow_xlsx_bytes)
    client.cookies.clear()
    register_verified("preview-stranger@example.com")
    resp = client.get(f"/api/sources/{sid}/sheets/48/preview")
    assert resp.status_code == 404
