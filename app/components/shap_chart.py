"""SHAP waterfall chart component — identical across all projects."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def shap_waterfall(shap_df: pd.DataFrame):
    if shap_df.empty:
        st.caption("No SHAP data available.")
        return
    colors = ["#E24B4A" if v > 0 else "#378ADD" for v in shap_df["shap_value"]]
    fig = go.Figure(go.Bar(
        x=shap_df["shap_value"],
        y=shap_df["feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in shap_df["shap_value"]],
        textposition="outside",
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=40, t=20, b=20),
        xaxis_title="SHAP value (impact on prediction)",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
