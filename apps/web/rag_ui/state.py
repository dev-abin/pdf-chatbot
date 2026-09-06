# frontend/rag_ui/state.py

import uuid
from typing import Any

import streamlit as st

ChatHistoryEntry = tuple[str, str, list[str]]
Thread = dict[str, Any]


def init_session_state() -> None:
    """
    Initialize Streamlit session_state with default keys.

    threads structure:
      {
        thread_id: {
          "title": str,
          "file": str | None,
          "document_id": int | None,
          "chat_history": [(q, a, sources)]
        }
      }
    """
    defaults: dict[str, Any] = {
        "access_token": None,
        "user": None,
        "threads": {},
        "current_thread_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if (
        st.session_state["access_token"]
        and st.session_state["current_thread_id"] is None
    ):
        create_new_thread()


def create_new_thread(title: str | None = None) -> None:
    threads: dict[str, Thread] = st.session_state["threads"]
    thread_id = str(uuid.uuid4())  # full UUID string

    if title is None:
        title = f"Chat {len(threads) + 1}"

    threads[thread_id] = {
        "title": title,
        "file": None,
        "document_id": None,
        "chat_history": [],
    }
    st.session_state["current_thread_id"] = thread_id


def get_current_thread() -> tuple[str | None, Thread | None]:
    threads: dict[str, Thread] = st.session_state["threads"]
    tid: str | None = st.session_state["current_thread_id"]
    if not tid or tid not in threads:
        return None, None
    return tid, threads[tid]


def switch_thread(thread_id: str) -> None:
    if thread_id in st.session_state["threads"]:
        st.session_state["current_thread_id"] = thread_id
