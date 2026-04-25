from __future__ import annotations

from fastapi.testclient import TestClient


def _upload_payload(name: str, data: bytes) -> dict:
    return {
        "files": {
            "file": (
                name,
                data,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
        "data": {"confirm_upload": "true"},
    }


def test_upload_and_list(
    client: TestClient, register_verified, sample_xlsx_bytes: bytes
) -> None:
    register_verified("alice@example.com")

    upload = client.post(
        "/api/sources", **_upload_payload("sample.xlsx", sample_xlsx_bytes)
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


def test_upload_rejects_non_excel(
    client: TestClient, register_verified
) -> None:
    register_verified("bob@example.com")

    resp = client.post(
        "/api/sources",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"confirm_upload": "true"},
    )
    assert resp.status_code == 400


def test_upload_rejects_xlsm(
    client: TestClient, register_verified, sample_xlsx_bytes: bytes
) -> None:
    register_verified("carol@example.com")

    resp = client.post(
        "/api/sources",
        files={
            "file": (
                "macro.xlsm",
                sample_xlsx_bytes,
                "application/vnd.ms-excel.sheet.macroEnabled.12",
            ),
        },
        data={"confirm_upload": "true"},
    )
    assert resp.status_code == 400


def test_upload_requires_confirm(
    client: TestClient, register_verified, sample_xlsx_bytes: bytes
) -> None:
    """UPLOAD-GATE-1: без `confirm_upload=true` загрузка не начинается."""

    register_verified("nogate@example.com")
    resp = client.post(
        "/api/sources",
        files={
            "file": (
                "x.xlsx",
                sample_xlsx_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
    )
    assert resp.status_code == 400
    assert "подтверждения" in resp.json()["detail"].lower()


def test_unverified_user_cannot_list_sources(
    client: TestClient, sample_xlsx_bytes: bytes
) -> None:
    client.post(
        "/api/auth/register",
        json={
            "email": "unverified@example.com",
            "password": "supersecret123",
            "accept_terms": True,
            "accept_pdn": True,
        },
    )
    resp = client.get("/api/sources")
    assert resp.status_code == 403
    upload = client.post(
        "/api/sources",
        files={
            "file": (
                "x.xlsx",
                sample_xlsx_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
        data={"confirm_upload": "true"},
    )
    assert upload.status_code == 403


def test_users_cannot_see_each_other_sources(
    client: TestClient, register_verified, sample_xlsx_bytes: bytes
) -> None:
    register_verified("one@example.com")
    upload = client.post(
        "/api/sources", **_upload_payload("one.xlsx", sample_xlsx_bytes)
    )
    source_id = upload.json()["id"]

    client.cookies.clear()
    register_verified("two@example.com")

    listed = client.get("/api/sources").json()
    assert listed == []

    resp = client.get(f"/api/sources/{source_id}")
    assert resp.status_code == 404
