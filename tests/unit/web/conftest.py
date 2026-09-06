# frontend/tests/conftest.py


import pytest
import streamlit as st


@pytest.fixture(autouse=True)
def reset_session_state() -> None:
    """
    Clear Streamlit session_state before every test.
    """
    st.session_state.clear()


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Provide dummy URLs so core.settings imports don’t explode.
    Adjust the env var names to match your settings.py.
    """
    monkeypatch.setenv("CHAT_API_URL", "http://testserver/chat/")
    monkeypatch.setenv("UPLOAD_FILE_URL", "http://testserver/upload-files/")
    monkeypatch.setenv("AUTH_LOGIN_URL", "http://testserver/auth/login")
    monkeypatch.setenv("AUTH_REGISTER_URL", "http://testserver/auth/register")
