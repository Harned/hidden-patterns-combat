"""Тесты мастера предобработки источника: lifecycle draft→ready и grid I/O."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _upload(
    client: TestClient,
    register_verified,
    email: str,
    data: bytes,
    name: str = "wiz.xlsx",
) -> int:
    register_verified(email)
    resp = client.post(
        "/api/sources",
        files={
            "file": (
                name,
                data,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
        data={"confirm_upload": "true"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_uploaded_source_starts_in_draft_state(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, register_verified, "draft@example.com", multirow_xlsx_bytes)
    card = client.get(f"/api/sources/{sid}").json()
    assert card["preparation_state"] == "draft"

    listed = client.get("/api/sources").json()
    assert listed[0]["preparation_state"] == "draft"


def test_analyze_blocked_for_draft(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, register_verified, "blocked@example.com", multirow_xlsx_bytes)
    resp = client.post(f"/api/sources/{sid}/analyze", params={"wait": "true"})
    assert resp.status_code == 400


def test_finalize_requires_mapping(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, register_verified, "no-mapping@example.com", multirow_xlsx_bytes)
    resp = client.post(f"/api/sources/{sid}/finalize")
    assert resp.status_code == 400


def test_finalize_promotes_to_ready_and_unblocks_analyze(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, register_verified, "wiz@example.com", multirow_xlsx_bytes)
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)

    resp = client.post(f"/api/sources/{sid}/finalize")
    assert resp.status_code == 200
    assert resp.json()["preparation_state"] == "ready"

    run = client.post(
        f"/api/sources/{sid}/analyze", params={"wait": "true"}
    )
    assert run.status_code == 200


def test_preflight_can_filter_sheets(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, register_verified, "filter@example.com", multirow_xlsx_bytes)

    sheets = client.get(f"/api/sources/{sid}/sheets").json()
    assert "48" in sheets["sheet_names"]

    resp = client.post(
        f"/api/sources/{sid}/preflight",
        json={"sheet_names": ["48"]},
    )
    assert resp.status_code == 200
    mapping = resp.json()["mapping"]
    assert list(mapping["sheets"].keys()) == ["48"]


def test_grid_read_returns_cells(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, register_verified, "grid@example.com", multirow_xlsx_bytes)
    resp = client.get(
        f"/api/sources/{sid}/sheets/48/grid",
        params={"start_row": 1, "start_col": 1, "n_rows": 6, "n_cols": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sheet"] == "48"
    assert body["n_rows"] == 6
    assert body["n_cols"] == 5
    assert body["cells"][0][0] == "ФИО борца"


def test_grid_edit_updates_cell(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, register_verified, "edit@example.com", multirow_xlsx_bytes)

    edit = client.put(
        f"/api/sources/{sid}/sheets/48/grid",
        json={"edits": [{"row": 4, "col": 1, "value": "Сидоров"}]},
    )
    assert edit.status_code == 200, edit.text
    assert edit.json()["applied"] == 1

    grid = client.get(
        f"/api/sources/{sid}/sheets/48/grid",
        params={"start_row": 4, "start_col": 1, "n_rows": 1, "n_cols": 1},
    ).json()
    assert grid["cells"][0][0] == "Сидоров"


def test_grid_edit_blocked_after_finalize(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, register_verified, "after@example.com", multirow_xlsx_bytes)
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)
    client.post(f"/api/sources/{sid}/finalize")

    resp = client.put(
        f"/api/sources/{sid}/sheets/48/grid",
        json={"edits": [{"row": 4, "col": 1, "value": "X"}]},
    )
    assert resp.status_code == 400


def test_remove_empty_rows_endpoint(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(client, register_verified, "empty@example.com", multirow_xlsx_bytes)

    # Сначала проставим пустую строку через grid PUT.
    # multirow_xlsx_bytes имеет 3 заголовочных строки (header_rows = [0,1,2])
    # и 3 data-строки (rows 4..6). Превратим row 5 в полностью пустую.
    client.put(
        f"/api/sources/{sid}/sheets/48/grid",
        json={
            "edits": [
                {"row": 5, "col": c, "value": None} for c in range(1, 11)
            ]
        },
    )
    resp = client.post(
        f"/api/sources/{sid}/sheets/48/remove-empty-rows",
        json={"header_rows": [0, 1, 2]},
    )
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 1


def test_athlete_forward_fill_suggestions_endpoint(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    """Очищаем ФИО в одной строке и ожидаем единственное предложение."""

    sid = _upload(
        client, register_verified, "athlete-suggest@example.com", multirow_xlsx_bytes
    )

    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)

    client.put(
        f"/api/sources/{sid}/sheets/48/grid",
        json={"edits": [{"row": 5, "col": 1, "value": None}]},
    )

    resp = client.get(
        f"/api/sources/{sid}/sheets/48/suggestions/athlete-forward-fill"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["athlete_column"]
    assert len(body["suggestions"]) == 1
    suggestion = body["suggestions"][0]
    assert suggestion["row"] == 5
    assert suggestion["col"] == 1
    assert suggestion["proposed"] == "Иванов"
    assert suggestion["source_row"] == 4


def test_athlete_forward_fill_suggestions_blocked_after_finalize(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    sid = _upload(
        client, register_verified, "athlete-final@example.com", multirow_xlsx_bytes
    )
    pre = client.post(f"/api/sources/{sid}/preflight").json()["mapping"]
    client.put(f"/api/sources/{sid}/mapping", json=pre)
    client.post(f"/api/sources/{sid}/finalize")

    resp = client.get(
        f"/api/sources/{sid}/sheets/48/suggestions/athlete-forward-fill"
    )
    assert resp.status_code == 400


def test_empty_rows_count_matches_remove(
    client: TestClient, register_verified, multirow_xlsx_bytes: bytes
) -> None:
    """Счётчик пустых строк должен совпадать с тем, что фактически удаляется."""

    sid = _upload(client, register_verified, "count@example.com", multirow_xlsx_bytes)

    # Сделаем одну пустую data-строку, как в test_remove_empty_rows_endpoint.
    client.put(
        f"/api/sources/{sid}/sheets/48/grid",
        json={
            "edits": [
                {"row": 5, "col": c, "value": None} for c in range(1, 11)
            ]
        },
    )

    pre = client.get(
        f"/api/sources/{sid}/sheets/48/empty-rows-count",
        params={"header_rows": "0,1,2"},
    )
    assert pre.status_code == 200
    expected = pre.json()["count"]
    assert expected == 1

    deleted = client.post(
        f"/api/sources/{sid}/sheets/48/remove-empty-rows",
        json={"header_rows": [0, 1, 2]},
    ).json()["deleted"]
    assert deleted == expected

    after = client.get(
        f"/api/sources/{sid}/sheets/48/empty-rows-count",
        params={"header_rows": "0,1,2"},
    ).json()["count"]
    assert after == 0
