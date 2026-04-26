"""E2E для HMM-ветки (TASK_SPEC_004) через HTTP."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _register_upload(
    client: TestClient, register_verified, email: str, data: bytes
) -> int:
    register_verified(email)
    resp = client.post(
        "/api/sources",
        files={
            "file": (
                "source.xlsx",
                data,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
        data={"confirm_upload": "true"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_analyze_returns_hmm_ready_on_dense_data(
    client: TestClient, register_verified, dense_hmm_xlsx_bytes: bytes
) -> None:
    sid = _register_upload(client, register_verified, "hmm-dense@example.com", dense_hmm_xlsx_bytes)
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)
    client.post(f"/api/sources/{sid}/finalize")
    run = client.post(
        f"/api/sources/{sid}/analyze", params={"wait": "true"}
    ).json()
    assert run["state"] == "done"
    assert run["status"] == "hmm_ready", run

    result = client.get(f"/api/sources/{sid}/result").json()["result"]
    assert result["status"] == "hmm_ready"
    assert result["hmm"] is not None
    labels = result["hmm"]["parameters"]["state_labels"]
    assert labels == ["маневрирование", "КФВ", "ВУП"]

    # Все Viterbi-состояния из доменного списка.
    assert result["hmm"]["trajectories"]
    for tr in result["hmm"]["trajectories"]:
        for s in tr["state_path"]:
            assert s in set(labels)


def test_analyze_blocks_hmm_on_thin_data(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _register_upload(client, register_verified, "hmm-thin@example.com", multirow_xlsx_bytes)
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)
    client.post(f"/api/sources/{sid}/finalize")
    run = client.post(
        f"/api/sources/{sid}/analyze", params={"wait": "true"}
    ).json()
    # Thin-data: guard должен заблокировать HMM.
    assert run["state"] == "done"
    assert run["status"] == "baseline_only"

    result = client.get(f"/api/sources/{sid}/result").json()["result"]
    assert result["status"] == "baseline_only"
    assert result["hmm"] is None
    codes = {w["code"] for w in result["warnings"]}
    assert "hmm.guards_failed" in codes


def test_analyze_rejects_unknown_mode(
    client: TestClient, register_verified, dense_hmm_xlsx_bytes: bytes
) -> None:
    sid = _register_upload(client, register_verified, "hmm-mode@example.com", dense_hmm_xlsx_bytes)
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)
    client.post(f"/api/sources/{sid}/finalize")
    resp = client.post(
        f"/api/sources/{sid}/analyze", params={"mode": "xyz", "wait": "true"}
    )
    assert resp.status_code == 400


def test_analyze_detailed_mode_on_dense_data(
    client: TestClient, register_verified, very_dense_xlsx_bytes: bytes
) -> None:
    sid = _register_upload(client, register_verified, "hmm-detailed@example.com", very_dense_xlsx_bytes)
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)
    client.post(f"/api/sources/{sid}/finalize")
    run = client.post(
        f"/api/sources/{sid}/analyze",
        params={"mode": "detailed", "wait": "true"},
    ).json()
    assert run["status"] == "hmm_ready", run

    result = client.get(f"/api/sources/{sid}/result").json()["result"]
    assert result["hmm"] is not None
    assert result["hmm"]["parameters"]["variant"] == "detailed_7state"
    assert result["hmm"]["parameters"]["n_states"] == 7
    labels = result["hmm"]["parameters"]["state_labels"]
    assert labels == [
        "маневры",
        "захваты",
        "хваты",
        "обхваты",
        "прихваты",
        "упоры",
        "ВУП",
    ]
