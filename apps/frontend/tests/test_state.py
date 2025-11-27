# frontend/tests/test_state.py

import streamlit as st
from rag_ui.state import (
    create_new_thread,
    get_current_thread,
    init_session_state,
    switch_thread,
)


def test_init_session_state_without_token_creates_no_thread():
    init_session_state()

    assert st.session_state["access_token"] is None
    assert st.session_state["user"] is None
    assert st.session_state["threads"] == {}
    assert st.session_state["current_thread_id"] is None


def test_init_session_state_with_token_creates_thread():
    st.session_state["access_token"] = "fake-token"

    init_session_state()

    threads = st.session_state["threads"]
    current_id = st.session_state["current_thread_id"]

    assert current_id is not None
    assert current_id in threads
    thread = threads[current_id]
    assert thread["title"] == "Chat 1"
    assert thread["file"] is None
    assert thread["document_id"] is None
    assert thread["chat_history"] == []


def test_create_new_thread_increments_title():
    init_session_state()
    st.session_state["access_token"] = "tok"

    # first thread
    create_new_thread()
    t1_id, t1 = get_current_thread()
    assert t1["title"] == "Chat 1"

    # second thread
    create_new_thread()
    t2_id, t2 = get_current_thread()
    assert t2["title"] == "Chat 2"
    assert t2_id != t1_id


def test_switch_thread_valid_and_invalid():
    init_session_state()
    st.session_state["access_token"] = "tok"
    create_new_thread()
    first_id, _ = get_current_thread()

    create_new_thread()
    second_id, _ = get_current_thread()

    # switch back to first
    switch_thread(first_id)
    assert st.session_state["current_thread_id"] == first_id

    # invalid id should not change current
    switch_thread("non-existent")
    assert st.session_state["current_thread_id"] == first_id
