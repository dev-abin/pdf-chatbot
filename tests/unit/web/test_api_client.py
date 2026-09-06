from typing import Any

import pytest
import streamlit as st
from rag_ui import api_client


class DummyResponse:
    def __init__(self, status_code: int, json_data: dict[str, Any] | None = None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self) -> dict[str, Any]:
        return self._json_data


@pytest.fixture
def mock_requests_post(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, Any]] = []

    def _fake_post(url, *args, **kwargs):
        calls.append({"url": url, "args": args, "kwargs": kwargs})
        # Default dummy 200 with empty JSON; tests override via monkeypatch
        return DummyResponse(200, {})

    monkeypatch.setattr(api_client, "requests", type("R", (), {"post": _fake_post}))
    return calls


def test_api_login_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url, *args, **kwargs):
        assert "auth/login" in url
        return DummyResponse(
            200, {"access_token": "tok", "token_type": "bearer", "user": {"email": "e"}}
        )

    monkeypatch.setattr(api_client, "requests", type("R", (), {"post": fake_post}))

    result = api_client.api_login("user@example.com", "pass")
    assert result is not None
    assert result["access_token"] == "tok"
    assert result["user"]["email"] == "e"


def test_api_login_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url, *args, **kwargs):
        return DummyResponse(401, {"detail": "Invalid credentials"})

    monkeypatch.setattr(api_client, "requests", type("R", (), {"post": fake_post}))

    result = api_client.api_login("user@example.com", "wrong")
    assert result is None  # st.error is called internally


def test_upload_file_for_thread_rejects_unsupported_extension() -> None:
    thread: dict[str, Any] = {"file": None, "document_id": None, "chat_history": []}

    class DummyFile:
        name = "image.jpg"

        def read(self, *args, **kwargs):
            return b"data"

    ok = api_client.upload_file_for_thread(DummyFile(), "thread-1", thread)
    assert ok is False
    assert thread["file"] is None
    assert thread["document_id"] is None


def test_upload_file_for_thread_success(monkeypatch: pytest.MonkeyPatch) -> None:
    thread: dict[str, Any] = {"file": None, "document_id": None, "chat_history": []}

    class DummyFile:
        name = "doc.txt"

        def read(self, *args, **kwargs):
            return b"hello"

    def fake_post(url, *args, **kwargs):
        files = kwargs.get("files")
        data = kwargs.get("data")
        assert "thread_id" in data and data["thread_id"] == "thread-xyz"
        assert "file" in files
        return DummyResponse(
            200,
            {
                "message": "ok",
                "document_id": 123,
                "filename": "doc.txt",
                "thread_id": "thread-xyz",
            },
        )

    monkeypatch.setattr(api_client, "requests", type("R", (), {"post": fake_post}))

    ok = api_client.upload_file_for_thread(DummyFile(), "thread-xyz", thread)
    assert ok is True
    assert thread["file"] == "doc.txt"
    assert thread["document_id"] == 123


def test_call_chat_backend_success(monkeypatch: pytest.MonkeyPatch) -> None:
    thread: dict[str, Any] = {
        "file": "doc.txt",
        "document_id": 42,
        "chat_history": [("hi", "hello", ["src1"])],
    }
    st.session_state["access_token"] = "tok"

    def fake_post(url, *args, **kwargs):
        body = kwargs.get("json")
        # history should be stripped to (q, a)
        assert body["chat_history"] == [("hi", "hello")]
        assert body["thread_id"] == "thread-abc"
        return DummyResponse(200, {"answer": "response", "sources": ["doc1"]})

    monkeypatch.setattr(api_client, "requests", type("R", (), {"post": fake_post}))

    result = api_client.call_chat_backend("question", "thread-abc", thread)
    assert result is not None
    assert result["answer"] == "response"
    assert result["sources"] == ["doc1"]
