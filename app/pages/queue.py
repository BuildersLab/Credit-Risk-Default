"""
Review Queue page.

Shows flagged records sorted by prediction score.
{{PERSONA}} clicks a row to open it in Case Review.

TODO: update column references and display fields for {{PROJECT_NAME}}.

Owner(s):
"""

import streamlit as st
import pandas as pd
from app.components.score_badge import score_badge


def render():
    st.title("Review Queue")
    st.caption("Flagged records sorted by prediction score. Click a row to open Case Review.")

    flagged = st.session_state.get("flagged_records", [])

    if not flagged:
        st.info("No flagged records in the current session. Use Batch Upload to score a file.")
        return

    df = pd.DataFrame(flagged)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total flagged", len(df))
    col2.metric("High score", len(df[df["score_tier"] == "High"]))
    # TODO: update "pred_prob" and value column for {{PROJECT_NAME}}
    col3.metric("Avg prediction score", f"{df['pred_prob'].mean():.1%}")

    st.divider()

    df_sorted = df.sort_values("pred_prob", ascending=False).reset_index(drop=True)

    for _, row in df_sorted.iterrows():
        with st.container():
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            # TODO: update display fields for {{PROJECT_NAME}}
            c1.markdown(f"Record `{row.get('id', _)}`")
            c2.markdown(f"{row.get('pred_prob', 0):.1%}")
            with c3:
                score_badge(row.get("score_tier", "Low"))
            with c4:
                if st.button("Review", key=f"review_{_}"):
                    st.session_state.current_record = row.to_dict()
                    st.session_state.gemini_explanation = None
                    st.session_state.chat_history = []
                    st.rerun()
        st.divider()
