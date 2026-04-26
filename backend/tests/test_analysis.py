from __future__ import annotations

from fastapi.testclient import TestClient


def _setup_source(client: TestClient, register_verified, email: str, data: bytes) -> int:
    register_verified(email)
    resp = client.post(
        "/api/sources",
        files={
            "file": (
                "sample.xlsx",
                data,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
        data={"confirm_upload": "true"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_analyze_and_fetch_result(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    source_id = _setup_source(
        client, register_verified, "alice@example.com", multirow_xlsx_bytes
    )
    pre = client.post(f"/api/sources/{source_id}/preflight").json()["mapping"]
    client.put(f"/api/sources/{source_id}/mapping", json=pre)
    client.post(f"/api/sources/{source_id}/finalize")

    run = client.post(f"/api/sources/{source_id}/analyze", params={"wait": "true"})
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["source_id"] == source_id
    assert body["state"] == "done"
    assert body["status"] == "baseline_only"

    result = client.get(f"/api/sources/{source_id}/result")
    assert result.status_code == 200
    payload = result.json()
    assert payload["status"] == "baseline_only"

    algo_result = payload["result"]
    assert algo_result["status"] == "baseline_only"
    # Инвариант: observations = ЗАП, никаких HMM-полей в MVP.
    assert algo_result["basic_statistics"]["hidden_group_totals"].get("ЗАП", 0) > 0
    for forbidden in ("viterbi_path", "hidden_states", "gamma"):
        assert forbidden not in algo_result


def test_result_absent_until_analyze(
    client: TestClient, register_verified, sample_xlsx_bytes: bytes
) -> None:
    source_id = _setup_source(client, register_verified, "bob@example.com", sample_xlsx_bytes)

    resp = client.get(f"/api/sources/{source_id}/result")
    assert resp.status_code == 404


def test_analyze_isolated_between_users(
    client: TestClient, register_verified, sample_xlsx_bytes: bytes
) -> None:
    source_id = _setup_source(client, register_verified, "owner@example.com", sample_xlsx_bytes)

    client.cookies.clear()
    register_verified("foe@example.com")

    resp = client.post(f"/api/sources/{source_id}/analyze")
    assert resp.status_code == 404

    resp2 = client.get(f"/api/sources/{source_id}/result")
    assert resp2.status_code == 404
