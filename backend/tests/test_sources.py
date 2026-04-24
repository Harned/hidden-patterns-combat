from __future__ import annotations

from fastapi.testclient import TestClient


def _register(client: TestClient, email: str) -> None:
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "password": "supersecret123"},
    )
    assert resp.status_code == 201


def test_upload_and_list(client: TestClient, sample_xlsx_bytes: bytes) -> None:
    _register(client, "alice@example.com")

    upload = client.post(
        "/api/sources",
        files={
            "file": (
                "sample.xlsx",
                sample_xlsx_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
    )
    assert upload.status_code == 201, upload.text
    source = upload.json()
    assert source["original_filename"] == "sample.xlsx"
    assert source["size_bytes"] == len(sample_xlsx_bytes)
    assert source["has_analysis"] is False

    listed = client.get("/api/sources")
    assert listed.status_code == 200
    items = listed.json()
    assert len(items) == 1
    assert items[0]["id"] == source["id"]


def test_upload_rejects_non_excel(client: TestClient) -> None:
    _register(client, "bob@example.com")

    resp = client.post(
        "/api/sources",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_rejects_xlsm(client: TestClient, sample_xlsx_bytes: bytes) -> None:
    _register(client, "carol@example.com")

    resp = client.post(
        "/api/sources",
        files={"file": ("macro.xlsm", sample_xlsx_bytes, "application/vnd.ms-excel.sheet.macroEnabled.12")},
    )
    assert resp.status_code == 400


def test_users_cannot_see_each_other_sources(
    client: TestClient, sample_xlsx_bytes: bytes
) -> None:
    _register(client, "one@example.com")
    upload = client.post(
        "/api/sources",
        files={"file": ("one.xlsx", sample_xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    source_id = upload.json()["id"]

    # Логинимся под другим пользователем
    client.cookies.clear()
    _register(client, "two@example.com")

    listed = client.get("/api/sources").json()
    assert listed == []

    resp = client.get(f"/api/sources/{source_id}")
    assert resp.status_code == 404
