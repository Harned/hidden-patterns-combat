from __future__ import annotations

from fastapi.testclient import TestClient


def _upload(client: TestClient, email: str, data: bytes, name: str = "m.xlsx") -> int:
    client.post(
        "/api/auth/register", json={"email": email, "password": "supersecret123"}
    )
    resp = client.post(
        "/api/sources",
        files={
            "file": (
                name,
                data,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_preflight_returns_roles(client: TestClient, multirow_xlsx_bytes: bytes) -> None:
    sid = _upload(client, "alice@example.com", multirow_xlsx_bytes)
    resp = client.post(f"/api/sources/{sid}/preflight")
    assert resp.status_code == 200, resp.text
    mapping = resp.json()["mapping"]
    assert mapping is not None
    sheet = mapping["sheets"]["48"]
    assert sheet["header_rows"] == [0, 1, 2]
    assert "ЗАП" in sheet["roles"]
    assert "маневрирование" in sheet["roles"]
    assert "КФВ" in sheet["roles"]
    assert "ВУП" in sheet["roles"]


def test_put_get_mapping_roundtrip(
    client: TestClient, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, "bob@example.com", multirow_xlsx_bytes)

    # Возьмём preflight как базу и сохраним его.
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    put = client.put(f"/api/sources/{sid}/mapping", json=pre)
    assert put.status_code == 200

    got = client.get(f"/api/sources/{sid}/mapping").json()["mapping"]
    assert got["sheets"] == pre["sheets"]

    # has_mapping отражается в карточке источника.
    card = client.get(f"/api/sources/{sid}").json()
    assert card["has_mapping"] is True


def test_delete_mapping(client: TestClient, multirow_xlsx_bytes: bytes) -> None:
    sid = _upload(client, "carol@example.com", multirow_xlsx_bytes)
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)

    resp = client.delete(f"/api/sources/{sid}/mapping")
    assert resp.status_code == 204
    card = client.get(f"/api/sources/{sid}").json()
    assert card["has_mapping"] is False


def test_put_mapping_rejects_invalid_payload(
    client: TestClient, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, "dave@example.com", multirow_xlsx_bytes)
    resp = client.put(
        f"/api/sources/{sid}/mapping", json={"sheets": "not-an-object"}
    )
    assert resp.status_code == 422


def test_analyze_with_saved_mapping_returns_baseline_only(
    client: TestClient, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, "eve@example.com", multirow_xlsx_bytes)
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)

    run = client.post(
        f"/api/sources/{sid}/analyze", params={"wait": "true"}
    ).json()
    assert run["status"] == "baseline_only"

    full = client.get(f"/api/sources/{sid}/result").json()
    result = full["result"]

    totals = result["basic_statistics"]["hidden_group_totals"]
    for g in ("ЗАП", "маневрирование", "КФВ", "ВУП"):
        assert totals.get(g, 0) > 0, f"group {g} must have data: {totals}"

    # Инвариант MVP: HMM-поля отсутствуют.
    for forbidden in ("viterbi_path", "hidden_states", "gamma"):
        assert forbidden not in result

    # Применённый mapping виден клиенту.
    assert result.get("applied_mapping")


def test_analyze_with_binary_zap_exposes_events_by_channel(
    client: TestClient, multirow_binary_xlsx_bytes: bytes
) -> None:
    sid = _upload(
        client, "binary@example.com", multirow_binary_xlsx_bytes, "binary.xlsx"
    )
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)

    run = client.post(
        f"/api/sources/{sid}/analyze", params={"wait": "true"}
    ).json()
    assert run["status"] == "baseline_only"

    full = client.get(f"/api/sources/{sid}/result").json()
    baseline = full["result"]["basic_statistics"]

    # ЗАП-события по каналам непустые и согласованы со schema.
    channels = baseline["zap_events_by_channel"]
    assert channels.get("Удержание") == 2
    assert channels.get("На руку") == 1
    assert channels.get("На ногу") == 1

    kinds = set(baseline["zap_column_kinds"].values())
    assert kinds <= {"binary", "count", "empty"}

    # Категорийных распределений ЗАП не должно быть на этой фикстуре.
    assert baseline["zap_value_counts"] == {}

    # HMM-полей всё ещё нет.
    for forbidden in ("viterbi_path", "hidden_states", "gamma"):
        assert forbidden not in full["result"]


def test_sheet_columns_endpoint_returns_all_columns(
    client: TestClient, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, "sc@example.com", multirow_xlsx_bytes)
    resp = client.get(
        f"/api/sources/{sid}/sheets/48/columns",
        params={"header_rows": "0,1,2"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["sheet"] == "48"
    assert data["header_rows"] == [0, 1, 2]
    names = [c["name"] for c in data["columns"]]
    assert "ФИО борца" in names
    assert "Баллы | ЗАП" in names

    # role_hint присутствует (может быть None).
    hints = {c["name"]: c.get("role_hint") for c in data["columns"]}
    assert hints["ФИО борца"] == "athlete"
    assert hints["Баллы | ЗАП"] == "ЗАП"


def test_sheet_columns_endpoint_rejects_bad_header_rows(
    client: TestClient, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, "sc2@example.com", multirow_xlsx_bytes)
    resp = client.get(
        f"/api/sources/{sid}/sheets/48/columns",
        params={"header_rows": "abc"},
    )
    assert resp.status_code == 400


def test_sheet_columns_endpoint_ownership(
    client: TestClient, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, "owner-sc@example.com", multirow_xlsx_bytes)
    client.cookies.clear()
    client.post(
        "/api/auth/register",
        json={"email": "stranger-sc@example.com", "password": "supersecret123"},
    )
    resp = client.get(f"/api/sources/{sid}/sheets/48/columns")
    assert resp.status_code == 404


def test_mapping_endpoints_enforce_ownership(
    client: TestClient, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, "owner@example.com", multirow_xlsx_bytes)

    client.cookies.clear()
    client.post(
        "/api/auth/register", json={"email": "stranger@example.com", "password": "supersecret123"}
    )
    for method, path in (
        ("post", f"/api/sources/{sid}/preflight"),
        ("get", f"/api/sources/{sid}/mapping"),
        ("put", f"/api/sources/{sid}/mapping"),
        ("delete", f"/api/sources/{sid}/mapping"),
    ):
        req = getattr(client, method)
        resp = req(path, json={}) if method in ("put",) else req(path)
        assert resp.status_code == 404, f"{method.upper()} {path} leaked"
