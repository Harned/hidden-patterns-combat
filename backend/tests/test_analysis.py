from __future__ import annotations

from fastapi.testclient import TestClient


def _setup_source(client: TestClient, email: str, data: bytes) -> int:
    client.post("/api/auth/register", json={"email": email, "password": "supersecret123"})
    resp = client.post(
        "/api/sources",
        files={
            "file": (
                "sample.xlsx",
                data,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
    )
    assert resp.status_code == 201
    return resp.json()["id"]


def test_analyze_and_fetch_result(client: TestClient, sample_xlsx_bytes: bytes) -> None:
    source_id = _setup_source(client, "alice@example.com", sample_xlsx_bytes)

    run = client.post(f"/api/sources/{source_id}/analyze")
    assert run.status_code == 201, run.text
    body = run.json()
    assert body["source_id"] == source_id
    assert body["status"] == "baseline_only"

    result = client.get(f"/api/sources/{source_id}/result")
    assert result.status_code == 200
    payload = result.json()
    assert payload["status"] == "baseline_only"

    algo_result = payload["result"]
    assert algo_result["status"] == "baseline_only"
    # Инвариант: observations = ЗАП, никаких HMM-полей в MVP.
    assert "ЗАП" in algo_result["detected_columns"]["detected_groups"]
    for forbidden in ("viterbi_path", "hidden_states", "gamma"):
        assert forbidden not in algo_result


def test_result_absent_until_analyze(client: TestClient, sample_xlsx_bytes: bytes) -> None:
    source_id = _setup_source(client, "bob@example.com", sample_xlsx_bytes)

    resp = client.get(f"/api/sources/{source_id}/result")
    assert resp.status_code == 404


def test_analyze_isolated_between_users(
    client: TestClient, sample_xlsx_bytes: bytes
) -> None:
    source_id = _setup_source(client, "owner@example.com", sample_xlsx_bytes)

    client.cookies.clear()
    client.post("/api/auth/register", json={"email": "foe@example.com", "password": "supersecret123"})

    resp = client.post(f"/api/sources/{source_id}/analyze")
    assert resp.status_code == 404

    resp2 = client.get(f"/api/sources/{source_id}/result")
    assert resp2.status_code == 404
