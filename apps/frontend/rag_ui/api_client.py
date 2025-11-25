import requests
import streamlit as st

from ..core.settings import (
    AUTH_LOGIN_URL,
    AUTH_REGISTER_URL,
    CHAT_API_URL,
    UPLOAD_FILE_URL,
)


def get_auth_headers():
    token = st.session_state.get("access_token")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def api_register(email: str, password: str):
    try:
        resp = requests.post(
            AUTH_REGISTER_URL,
            json={"email": email, "password": password},
            timeout=10,
        )
    except requests.RequestException as e:
        st.error(f"Could not reach auth service: {e}")
        return None

    if resp.status_code != 200 and resp.status_code != 201:
        try:
            data = resp.json()
            detail = data.get(
                "detail", f"Registration failed (status {resp.status_code})"
            )
        except Exception:
            detail = f"Registration failed (status {resp.status_code})"
        st.error(detail)
        return None

    return resp.json()


def api_login(email: str, password: str):
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
        try:
            data = resp.json()
            detail = data.get("detail", f"Login failed (status {resp.status_code})")
        except Exception:
            detail = f"Login failed (status {resp.status_code})"
        st.error(detail)
        return None

    return resp.json()


def upload_file_for_thread(uploaded_file, thread):
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

    try:
        resp = requests.post(
            UPLOAD_FILE_URL,
            files=files,
            headers=get_auth_headers(),
            timeout=60,
        )
    except requests.RequestException as e:
        st.error(f"Upload failed: {e}")
        return False

    if resp.status_code == 200:
        thread["file"] = uploaded_file.name
        st.success(f"File `{uploaded_file.name}` uploaded and indexed successfully.")
        return True
    else:
        try:
            data = resp.json()
            detail = data.get(
                "detail", f"Failed to upload file (status {resp.status_code})"
            )
        except Exception:
            detail = f"Failed to upload file (status {resp.status_code})"
        st.error(detail)
        return False


def call_chat_backend(question: str, thread):
    # Convert internal chat_history [(q, a, sources)] → [(q, a)]
    formatted_history = [(q, a) for (q, a, _) in thread["chat_history"]]

    try:
        resp = requests.post(
            CHAT_API_URL,
            json={"question": question, "chat_history": formatted_history},
            headers=get_auth_headers(),
            timeout=120,
        )
    except requests.RequestException as e:
        st.error(f"Chat request failed: {e}")
        return None

    if resp.status_code != 200:
        try:
            data = resp.json()
            detail = data.get(
                "detail", f"Chat request failed (status {resp.status_code})"
            )
        except Exception:
            detail = f"Chat request failed (status {resp.status_code})"
        st.error(detail)
        return None

    return resp.json()
