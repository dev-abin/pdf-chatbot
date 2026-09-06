import streamlit as st

from .api_client import api_login, api_register
from .state import create_new_thread


def login_view():
    st.title("Login")

    # controls whether to show register form
    if "show_register" not in st.session_state:
        st.session_state["show_register"] = False

    if st.session_state["show_register"]:
        register_view()
        return

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if not email or not password:
            st.warning("Please enter both email and password.")
            return

        data = api_login(email, password)
        if data is None:
            return

        access_token = data.get("access_token")
        user = data.get("user")

        if not access_token or not user:
            st.error("Invalid response from auth server.")
            return

        st.session_state["access_token"] = access_token
        st.session_state["user"] = user
        st.session_state["threads"] = {}
        st.session_state["current_thread_id"] = None

        create_new_thread()
        st.rerun()

    st.markdown("---")
    if st.button("Create a new account"):
        st.session_state["show_register"] = True
        st.rerun()


def register_view():
    st.title("Register")

    with st.form("register_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")

    if submitted:
        if not email or not password or not confirm:
            st.warning("Please fill all fields.")
            return
        if password != confirm:
            st.warning("Passwords do not match.")
            return

        resp = api_register(email, password)
        if resp is None:
            return

        st.success("Account created. Please login.")
        st.session_state["show_register"] = False
        st.rerun()

    st.markdown("---")
    if st.button("Back to Login"):
        st.session_state["show_register"] = False
        st.rerun()


def logout():
    st.session_state["access_token"] = None
    st.session_state["user"] = None
    st.session_state["threads"] = {}
    st.session_state["current_thread_id"] = None
    st.rerun()
