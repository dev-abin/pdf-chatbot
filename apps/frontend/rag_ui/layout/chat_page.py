import streamlit as st

from ..api_client import call_chat_backend
from ..state import get_current_thread


def chat_layout():
    st.title("Document Chatbot")

    current_id, thread = get_current_thread()
    if thread is None:
        st.info("No active conversation. Create a new chat from the sidebar.")
        return

    if thread["file"]:
        st.caption(f"Chatting with file: `{thread['file']}`")
    else:
        st.caption(
            "No file uploaded yet. You can still ask questions, or upload a file in the sidebar."
        )

    # Render chat history
    for q, a, sources in thread["chat_history"]:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
            if sources:
                st.markdown(f"**Sources:** {', '.join(sources)}")

    # New user input
    user_input = st.chat_input("Ask about your document or general question")
    if user_input:
        # Show user's message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = call_chat_backend(user_input, thread)

            if result is None:
                return

            assistant_response = result.get("answer", "")
            sources = result.get("sources", [])

            st.markdown(assistant_response)
            if sources:
                st.markdown(f"**Sources:** {', '.join(sources)}")

        # Persist to thread history
        thread["chat_history"].append((user_input, assistant_response, sources))
