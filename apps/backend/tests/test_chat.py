# apps/backend/tests/test_chat.py

from io import BytesIO

from fastapi import status


def register_and_login(client) -> str:
    email = "chatuser@example.com"
    password = "strongpass"

    client.post("/auth/register", json={"email": email, "password": password})
    resp = client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == status.HTTP_200_OK
    return resp.json()["access_token"]


def test_chat_requires_auth(client):
    payload = {
        "question": "Hello?",
        "chat_history": [],
        "thread_id": "thread-1",
    }
    resp = client.post("/chat/", json=payload)
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED


def test_chat_no_vectorstore(client):
    """
    When no vectorstore exists yet, /chat/ should return 404 with a clear message.
    """
    token = register_and_login(client)
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "question": "What is inside my docs?",
        "chat_history": [],
        "thread_id": "thread-1",
    }
    resp = client.post("/chat/", headers=headers, json=payload)
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert "Please upload a document first" in resp.text


def test_chat_with_uploaded_doc_and_mocked_llm(client, tmp_path, monkeypatch):
    """
    Full flow with:
      - temp VECTOR_DIR / FILE_DIR
      - upload a TXT
      - mock get_chat_llm so answer is deterministic
    """
    token = register_and_login(client)
    headers = {"Authorization": f"Bearer {token}"}

    # Patch dirs
    from app.core import settings

    settings.VECTOR_DIR = tmp_path / "vector"
    settings.FILE_DIR = tmp_path / "files"
    settings.VECTOR_DIR.mkdir(exist_ok=True, parents=True)
    settings.FILE_DIR.mkdir(exist_ok=True, parents=True)

    # 1) Upload
    file_content = b"pytest is a framework for writing tests in python"
    files = {"file": ("test.txt", BytesIO(file_content), "text/plain")}

    resp = client.post(
        "/upload-files/",
        headers=headers,
        files=files,
        data={"thread_id": "thread-abc"},
    )
    assert resp.status_code == status.HTTP_200_OK

    # 2) Monkeypatch LLM to return a fixed response
    from app.core import llm_client

    class DummyLLM:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, prompt):
            # For ChatPromptTemplate | LLM, prompt will often be a dict or string.
            # We only care that content gets through.
            return "This is a mocked answer."

        # If you are using LangChain chat models, they may call .invoke() with a dict
        # and expect an object with `.content`. You can adjust accordingly.

    def fake_get_chat_llm(*args, **kwargs):
        return DummyLLM()

    monkeypatch.setattr(llm_client, "get_chat_llm", fake_get_chat_llm)

    # 3) Call chat
    payload = {
        "question": "What is pytest?",
        "chat_history": [],
        "thread_id": "thread-abc",
    }

    resp = client.post("/chat/", headers=headers, json=payload)
    assert resp.status_code == status.HTTP_200_OK

    data = resp.json()
    assert "answer" in data
    assert "mocked answer" in data["answer"].lower()
