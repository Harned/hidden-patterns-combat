from __future__ import annotations

from fastapi.testclient import TestClient


def test_register_sets_session_cookie(client: TestClient) -> None:
    resp = client.post(
        "/api/auth/register",
        json={"email": "alice@example.com", "password": "supersecret123"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["email"] == "alice@example.com"
    assert "hpc_session" in resp.cookies


def test_register_duplicate_email(client: TestClient) -> None:
    payload = {"email": "bob@example.com", "password": "supersecret123"}
    assert client.post("/api/auth/register", json=payload).status_code == 201
    r2 = client.post("/api/auth/register", json=payload)
    assert r2.status_code == 409


def test_login_and_me(client: TestClient) -> None:
    client.post(
        "/api/auth/register",
        json={"email": "carol@example.com", "password": "supersecret123"},
    )
    client.cookies.clear()

    login = client.post(
        "/api/auth/login",
        json={"email": "carol@example.com", "password": "supersecret123"},
    )
    assert login.status_code == 200
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "carol@example.com"


def test_me_requires_auth(client: TestClient) -> None:
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401


def test_login_wrong_password(client: TestClient) -> None:
    client.post(
        "/api/auth/register",
        json={"email": "dave@example.com", "password": "supersecret123"},
    )
    client.cookies.clear()
    resp = client.post(
        "/api/auth/login",
        json={"email": "dave@example.com", "password": "wrong-password"},
    )
    assert resp.status_code == 401


def test_logout_clears_session(client: TestClient) -> None:
    client.post(
        "/api/auth/register",
        json={"email": "eve@example.com", "password": "supersecret123"},
    )
    assert client.get("/api/auth/me").status_code == 200

    logout = client.post("/api/auth/logout")
    assert logout.status_code == 204
    client.cookies.clear()
    assert client.get("/api/auth/me").status_code == 401
