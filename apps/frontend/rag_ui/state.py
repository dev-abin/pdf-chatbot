import uuid

import streamlit as st


def init_session_state():
    defaults = {
        "access_token": None,
        "user": None,
        # threads: {thread_id: {"title": str, "file": str|None, "chat_history": [(q, a, sources)]}}
        "threads": {},
        "current_thread_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Ensure we always have at least one thread after login
    if (
        st.session_state["access_token"]
        and st.session_state["current_thread_id"] is None
    ):
        create_new_thread()


def create_new_thread(title: str | None = None):
    threads = st.session_state["threads"]
    thread_id = str(uuid.uuid4())[:8]  # short id for display/keys

    if title is None:
        thread_num = len(threads) + 1
        title = f"Chat {thread_num}"

    threads[thread_id] = {
        "title": title,
        "file": None,
        "chat_history": [],  # list of (question, answer, sources)
    }
    st.session_state["current_thread_id"] = thread_id


def get_current_thread():
    threads = st.session_state["threads"]
    tid = st.session_state["current_thread_id"]
    if not tid or tid not in threads:
        return None, None
    return tid, threads[tid]


def switch_thread(thread_id: str):
    if thread_id in st.session_state["threads"]:
        st.session_state["current_thread_id"] = thread_id
