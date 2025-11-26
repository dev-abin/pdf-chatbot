# frontend/rag_ui/api_client.py

from typing import Any

import requests
import streamlit as st

from ..core.settings import (
    AUTH_LOGIN_URL,
    AUTH_REGISTER_URL,
    CHAT_API_URL,
    UPLOAD_FILE_URL,
)


def get_auth_headers() -> dict[str, str]:
    token: str | None = st.session_state.get("access_token")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _extract_detail(resp: requests.Response, fallback: str) -> str:
    try:
        data = resp.json()
        return str(data.get("detail", fallback))
    except Exception:
        return fallback


def api_register(email: str, password: str) -> dict[str, Any] | None:
    try:
        resp = requests.post(
            AUTH_REGISTER_URL,
            json={"email": email, "password": password},
            timeout=10,
        )
    except requests.RequestException as e:
        st.error(f"Could not reach auth service: {e}")
        return None

    if resp.status_code not in (200, 201):
        detail = _extract_detail(
            resp, f"Registration failed (status {resp.status_code})"
        )
        st.error(detail)
        return None

    return resp.json()


def api_login(email: str, password: str) -> dict[str, Any] | None:
    try:
        resp = requests.post(
            AUTH_LOGIN_URL,
            json={"email": email, "password": password},
            timeout=10,
        )
    except requests.RequestException as e:
        st.error(f"Could not reach auth service: {e}")
        return None

    if resp.status_code != 200:
        detail = _extract_detail(resp, f"Login failed (status {resp.status_code})")
        st.error(detail)
        return None

    return resp.json()


def upload_file_for_thread(
    uploaded_file, thread_id: str, thread: dict[str, Any]
) -> bool:
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        mime_type = "application/pdf"
    elif filename.endswith(".docx"):
        mime_type = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    elif filename.endswith(".txt"):
        mime_type = "text/plain"
    else:
        st.error("Unsupported file type. Only PDF, DOCX, TXT are allowed.")
        return False

    files = {"file": (uploaded_file.name, uploaded_file, mime_type)}
    data = {"thread_id": thread_id}

    try:
        resp = requests.post(
            UPLOAD_FILE_URL,
            data=data,
            files=files,
            headers=get_auth_headers(),
            timeout=60,
        )
    except requests.RequestException as e:
        st.error(f"Upload failed: {e}")
        return False

    if resp.status_code == 200:
        try:
            data = resp.json()
        except Exception:
            data = {}

        thread["file"] = uploaded_file.name
        if "document_id" in data:
            thread["document_id"] = data["document_id"]

        msg = data.get(
            "message", f"File `{uploaded_file.name}` uploaded and indexed successfully."
        )
        st.success(msg)
        return True

    detail = _extract_detail(resp, f"Failed to upload file (status {resp.status_code})")
    st.error(detail)
    return False


def call_chat_backend(
    question: str, thread_id: str, thread: dict[str, Any]
) -> dict[str, Any] | None:
    """
    Call the /chat/ endpoint with question, chat_history, and thread_id.
    """
    formatted_history = [(q, a) for (q, a, _) in thread["chat_history"]]

    payload = {
        "question": question,
        "chat_history": formatted_history,
        "thread_id": thread_id,
    }

    try:
        resp = requests.post(
            CHAT_API_URL,
            json=payload,
            headers=get_auth_headers(),
            timeout=120,
        )
    except requests.RequestException as e:
        st.error(f"Chat request failed: {e}")
        return None

    if resp.status_code != 200:
        detail = _extract_detail(
            resp, f"Chat request failed (status {resp.status_code})"
        )
        st.error(detail)
        return None

    try:
        return resp.json()
    except Exception as e:
        st.error(f"Invalid response from chat backend: {e}")
        return None
