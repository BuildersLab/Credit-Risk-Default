"""
Credit Risk Default dashboard — Streamlit entrypoint.

Owner(s):
"""

import streamlit as st
import joblib
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Credit Risk Default | NorthBay Bank TBD",
    page_icon="📊",  # TODO: update icon to match project
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    model_path = Path(os.getenv("MODEL_DIR", "./models")) / "champion.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def init_session_state():
    defaults = {
        "model": load_model(),
        "flagged_records": [],
        "user_actions": {},
        "chat_history": [],
        "current_record": None,
        "session_metrics": {
            "total_scored": 0,
            "total_flagged": 0,
            "value_caught": 0.0,
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

with st.sidebar:
    st.markdown("### Credit Risk Default")
    st.markdown("**NorthBay Bank TBD**")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Review Queue", "Case Review", "Batch Upload", "Summary and Report"],
        label_visibility="collapsed",
    )

    st.divider()
    if st.session_state.model is None:
        st.warning("Model not loaded. Run `make train` first.")
    else:
        st.success("Model loaded")

    if os.getenv("GEMINI_API_KEY"):
        st.success("Gemini API connected")
    else:
        st.warning("Gemini API key not set")

if page == "Review Queue":
    from app.pages.queue import render
    render()
elif page == "Case Review":
    from app.pages.case_review import render
    render()
elif page == "Batch Upload":
    from app.pages.batch_upload import render
    render()
elif page == "Summary and Report":
    from app.pages.summary import render
    render()
