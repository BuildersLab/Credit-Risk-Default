"""
Batch Upload page.

Owner(s):
"""
import streamlit as st
import pandas as pd


def render():
    st.title("Batch Upload")
    st.caption("Upload a CSV of raw records. The model scores them in real time.")

    model = st.session_state.get("model")
    if model is None:
        st.error("Model not loaded. Run `make train` first.")
        return

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df):,} records.")

            with st.expander("Preview", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

            if st.button("Score records", type="primary"):
                from src.predict import score_batch
                with st.spinner(f"Scoring {len(df):,} records..."):
                    scored = score_batch(uploaded, model)

                flagged = scored[scored["pred_flag"] == 1]
                st.session_state.flagged_records = flagged.to_dict("records")

                c1, c2, c3 = st.columns(3)
                c1.metric("Total scored", len(scored))
                c2.metric("Flagged", len(flagged))
                c3.metric("Flag rate", f"{len(flagged)/len(scored):.2%}")

                st.dataframe(
                    scored[["pred_prob", "score_tier", "pred_flag"]].head(50),
                    use_container_width=True,
                )

                st.download_button(
                    "Download scored results",
                    data=scored.to_csv(index=False).encode(),
                    file_name="{{PROJECT_SLUG}}_scored.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error: {e}")
