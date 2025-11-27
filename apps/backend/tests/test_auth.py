from fastapi import status


def test_register_and_login_success(client):
    # Register
    resp = client.post(
        "/auth/register",
        json={"email": "user@example.com", "password": "strongpass"},
    )
    assert resp.status_code in (status.HTTP_200_OK, status.HTTP_201_CREATED)
    data = resp.json()
    assert "user" in data
    assert data["user"]["email"] == "user@example.com"

    # Login
    resp = client.post(
        "/auth/login",
        json={"email": "user@example.com", "password": "strongpass"},
    )
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "user@example.com"


def test_register_duplicate_email_fails(client):
    client.post(
        "/auth/register",
        json={"email": "dup@example.com", "password": "p1"},
    )
    resp = client.post(
        "/auth/register",
        json={"email": "dup@example.com", "password": "p2"},
    )
    # Expect 400 or 409 depending on your implementation; adjust if needed
    assert resp.status_code in (status.HTTP_400_BAD_REQUEST, status.HTTP_409_CONFLICT)


def test_login_wrong_password(client):
    client.post(
        "/auth/register",
        json={"email": "user2@example.com", "password": "correct"},
    )
    resp = client.post(
        "/auth/login",
        json={"email": "user2@example.com", "password": "wrong"},
    )
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
