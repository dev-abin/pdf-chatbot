# frontend/rag_ui/layout/chat_page.py

import streamlit as st

from ..api_client import call_chat_backend
from ..state import get_current_thread


def _render_sources(sources: list[str]) -> None:
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                st.markdown(f"- [{src}]({src})")
            else:
                st.markdown(f"- {src}")


def _render_chat_stats(thread: dict) -> None:
    total_sources = sum(len(sources) for _, _, sources in thread["chat_history"])
    columns = st.columns(3)
    columns[0].metric("Turns", len(thread["chat_history"]))
    columns[1].metric("Sources cited", total_sources)
    with columns[2]:
        st.markdown("**Active file**")
        if thread["file"]:
            st.code(thread["file"], language=None)
        else:
            st.caption("None")


def chat_layout() -> None:
    st.title("Document Chatbot")
    st.caption("Ask questions about your documents or explore general topics.")

    current_id, thread = get_current_thread()
    if thread is None or current_id is None:
        st.info("No active conversation. Create a new chat from the sidebar.")
        return

    _render_chat_stats(thread)
    st.divider()

    if thread["file"]:
        st.caption(f"Chatting with file: `{thread['file']}`")
    else:
        st.caption(
            "No file uploaded yet. You can still ask questions, or upload a file in the sidebar."
        )

    if not thread["chat_history"]:
        st.info("Try one of these prompts to get started:")
        st.markdown(
            "- Summarize the document in three bullet points.\n"
            "- What are the key dates or milestones?\n"
            "- Highlight any risks, assumptions, or open questions."
        )

    for q, a, sources in thread["chat_history"]:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
            _render_sources(sources)

    user_input = st.chat_input("Ask about your document or a general question")
    if not user_input:
        return

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = call_chat_backend(user_input, current_id, thread)

        if result is None:
            return

        assistant_response: str = result.get("answer", "")
        sources = result.get("sources", [])

        if not assistant_response:
            st.error("Chat backend returned an empty answer.")
            return

        st.markdown(assistant_response)
        _render_sources(sources)

    thread["chat_history"].append((user_input, assistant_response, sources))
