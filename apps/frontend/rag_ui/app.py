import streamlit as st

from .auth import login_view
from .layout.chat_page import chat_layout
from .layout.sidebar import sidebar_layout
from .state import init_session_state


def app():
    st.set_page_config(page_title="RAG Chat", layout="wide")
    init_session_state()

    if not st.session_state.get("access_token"):
        login_view()
        return

    sidebar_layout()
    chat_layout()
