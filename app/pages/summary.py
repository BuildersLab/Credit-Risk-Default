"""
Summary and Report page.

Owner(s):
"""
import streamlit as st


def render():
    st.title("Summary and Report")

    flagged = st.session_state.get("flagged_records", [])
    actions = st.session_state.get("user_actions", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Flagged", len(flagged))
    c2.metric("Actioned", len(actions))
    # TODO: add a business value metric for {{PROJECT_NAME}}
    c3.metric("Avg risk score", f"{sum(r.get('pred_prob',0) for r in flagged)/max(len(flagged),1):.1%}")

    st.divider()

    st.subheader("Patterns in this batch")
    st.caption("Powered by Gemini AI.")

    if not flagged:
        st.info("No flagged records. Upload a batch first.")
    else:
        if st.button("Detect patterns", type="primary"):
            try:
                from src.gemini import detect_patterns
                with st.spinner("Analysing patterns..."):
                    patterns = detect_patterns(flagged)
                st.session_state["detected_patterns"] = patterns
            except Exception as e:
                st.error(f"Pattern detection failed: {e}")

        if st.session_state.get("detected_patterns"):
            st.markdown(st.session_state.detected_patterns)

    st.divider()

    st.subheader("Stakeholder report")
    st.caption("Executive summary for {{COMPANY_NAME}} leadership.")

    if st.button("Generate report", type="primary"):
        try:
            from src.gemini import generate_report
            # TODO: populate metrics from actual session data for {{PROJECT_NAME}}
            metrics = st.session_state.get("session_metrics", {})
            with st.spinner("Generating report..."):
                report = generate_report(
                    metrics=metrics,
                    flagged_records=flagged,
                    date_range="today's batch",
                )
            st.session_state["stakeholder_report"] = report
        except Exception as e:
            st.error(f"Report generation failed: {e}")

    if st.session_state.get("stakeholder_report"):
        st.markdown(st.session_state.stakeholder_report)
        st.download_button(
            "Download report",
            data=st.session_state.stakeholder_report,
            file_name="{{PROJECT_SLUG}}_report.txt",
            mime="text/plain",
        )
