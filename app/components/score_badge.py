"""Score tier badge component — identical across all projects."""
import streamlit as st


def score_badge(tier: str):
    colors = {"High": "red", "Medium": "orange", "Low": "green"}
    color = colors.get(tier, "gray")
    st.markdown(
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:12px;font-weight:500">{tier}</span>',
        unsafe_allow_html=True,
    )
