# frontend/tests/test_layout_smoke.py

import streamlit as st
from rag_ui.layout.chat_page import chat_layout
from rag_ui.layout.sidebar import sidebar_layout
from rag_ui.state import create_new_thread, init_session_state


def test_sidebar_layout_no_threads():
    init_session_state()
    st.session_state["access_token"] = "tok"
    sidebar_layout()  # should not raise


def test_chat_layout_with_empty_thread(monkeypatch):
    init_session_state()
    st.session_state["access_token"] = "tok"
    create_new_thread()

    # Stub chat_input to immediately return None so layout exits quickly
    from rag_ui.layout import chat_page

    def fake_chat_input(*args, **kwargs):
        return None

    monkeypatch.setattr(chat_page.st, "chat_input", fake_chat_input)

    chat_layout()  # should not raise


def test_chat_layout_with_message(monkeypatch):
    init_session_state()
    st.session_state["access_token"] = "tok"
    create_new_thread()

    from rag_ui import api_client
    from rag_ui.layout import chat_page

    # chat_input returns a single question once
    called = {"count": 0}

    def fake_chat_input(*args, **kwargs):
        if called["count"] == 0:
            called["count"] += 1
            return "Hello?"
        return None

    monkeypatch.setattr(chat_page.st, "chat_input", fake_chat_input)

    # Mock backend call to avoid real HTTP
    def fake_call_chat_backend(question, thread_id, thread):
        return {"answer": "Hi there!", "sources": ["dummy"]}

    monkeypatch.setattr(api_client, "call_chat_backend", fake_call_chat_backend)

    chat_layout()  # should run through one message without raising
