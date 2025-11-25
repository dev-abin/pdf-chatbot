import streamlit as st

from ..api_client import upload_file_for_thread
from ..auth import logout
from ..state import create_new_thread, get_current_thread, switch_thread


def sidebar_layout():
    user = st.session_state.get("user")

    st.sidebar.title("RAG Chat")

    if user:
        st.sidebar.markdown(f"**User:** {user.get('email', 'Unknown')}")

    if st.sidebar.button("New Chat", use_container_width=True):
        create_new_thread()
        st.rerun()

    threads = st.session_state["threads"]
    if not threads:
        st.sidebar.info("No conversations yet.")
    else:
        options = list(threads.keys())
        current_id = st.session_state["current_thread_id"]

        def fmt_thread(tid: str) -> str:
            t = threads[tid]
            label = t["title"]
            if t["file"]:
                label += f" · {t['file']}"
            return label

        selected = st.sidebar.radio(
            "Past conversations",
            options=options,
            index=options.index(current_id) if current_id in options else 0,
            format_func=fmt_thread,
        )
        if selected != current_id:
            switch_thread(selected)
            st.rerun()

    st.sidebar.markdown("---")

    # File upload for current thread
    current_id, thread = get_current_thread()
    if thread is None:
        st.sidebar.info("No active conversation.")
        return

    if thread["file"]:
        st.sidebar.success(f"Current file: `{thread['file']}`")
    else:
        st.sidebar.info("No file uploaded for this chat yet.")

    uploaded_file = st.sidebar.file_uploader(
        "Upload a file for this chat",
        type=["pdf", "docx", "txt"],
    )

    if uploaded_file:
        if thread["file"] == uploaded_file.name:
            st.sidebar.info(f"`{uploaded_file.name}` already used for this chat.")
        else:
            with st.sidebar.status("Indexing file…", expanded=True) as status_box:
                ok = upload_file_for_thread(uploaded_file, thread)
                if ok:
                    status_box.update(
                        label="✅ File indexed", state="complete", expanded=False
                    )
                    st.rerun()
                else:
                    status_box.update(
                        label="❌ Failed to index file", state="error", expanded=False
                    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        logout()
