# apps/backend/tests/test_upload.py

from io import BytesIO

from fastapi import status


def register_and_login(client) -> str:
    email = "uploaduser@example.com"
    password = "strongpass"

    client.post("/auth/register", json={"email": email, "password": password})
    resp = client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == status.HTTP_200_OK
    return resp.json()["access_token"]


def test_upload_requires_auth(client):
    file_content = b"dummy text"
    files = {"file": ("test.txt", BytesIO(file_content), "text/plain")}
    resp = client.post("/upload-files/", files=files, data={"thread_id": "t-1"})
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED


def test_upload_rejects_unsupported_extension(client):
    token = register_and_login(client)
    headers = {"Authorization": f"Bearer {token}"}

    file_content = b"%PDF-1.4"
    files = {"file": ("image.jpg", BytesIO(file_content), "image/jpeg")}

    resp = client.post(
        "/upload-files/",
        headers=headers,
        files=files,
        data={"thread_id": "thread-xyz"},
    )
    assert resp.status_code == status.HTTP_400_BAD_REQUEST
    assert "File must be a PDF/DOCX or TXT file" in resp.text


def test_upload_txt_success(client, tmp_path, monkeypatch):
    """
    End-to-end test of upload for a simple TXT file.
    We monkeypatch VECTOR_DIR and FILE_DIR to temp dirs if needed.
    """
    token = register_and_login(client)
    headers = {"Authorization": f"Bearer {token}"}

    # If your app reads VECTOR_DIR/FILE_DIR from settings at import time,
    # you may need to monkeypatch those modules before importing.
    # Example if settings.VECTOR_DIR is a Path:
    from app.core import settings

    settings.VECTOR_DIR = tmp_path / "vector"
    settings.FILE_DIR = tmp_path / "files"
    settings.VECTOR_DIR.mkdir(exist_ok=True, parents=True)
    settings.FILE_DIR.mkdir(exist_ok=True, parents=True)

    file_content = b"hello world\nthis is a test"
    files = {"file": ("test.txt", BytesIO(file_content), "text/plain")}

    resp = client.post(
        "/upload-files/",
        headers=headers,
        files=files,
        data={"thread_id": "thread-xyz"},
    )
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert "Successfully processed" in data["message"]
    assert data["filename"] == "test.txt"
    assert data["thread_id"] == "thread-xyz"
    assert isinstance(data["document_id"], int)
