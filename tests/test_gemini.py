"""
Tests for src/gemini.py and src/prompts.py

These tests verify prompt formatting and error handling.
They do not make live API calls — CI runs without a Gemini key.

TODO: update prompt field assertions for {{PROJECT_NAME}}.
"""

import pytest
from src.prompts import explanation_prompt, pattern_detection_prompt, report_prompt


def test_explanation_prompt_is_string():
    import pandas as pd
    shap_df = pd.DataFrame({
        "feature": ["feature_a", "feature_b"],
        "feature_value": [1.0, 2.0],
        "direction": ["increases risk", "decreases risk"],
    })
    # TODO: update with actual fields for {{PROJECT_NAME}}
    prompt = explanation_prompt(record={"amount": 100}, shap_df=shap_df)
    assert isinstance(prompt, str)
    assert len(prompt) > 50


def test_explanation_prompt_no_shap_jargon():
    import pandas as pd
    shap_df = pd.DataFrame({
        "feature": ["feature_a"],
        "feature_value": [1.0],
        "direction": ["increases risk"],
    })
    prompt = explanation_prompt(record={}, shap_df=shap_df)
    assert "shap_value" not in prompt.lower()
    assert "shapley" not in prompt.lower()


def test_pattern_detection_prompt_includes_count():
    records = [{"amount": 100}, {"amount": 200}, {"amount": 300}]
    prompt = pattern_detection_prompt(records)
    assert "3" in prompt


def test_report_prompt_includes_date_range():
    metrics = {}
    prompt = report_prompt(metrics, "No patterns found.", "January 2020")
    assert "January 2020" in prompt


def test_gemini_raises_without_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    import importlib
    import src.gemini as g
    importlib.reload(g)
    import pandas as pd
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        g.explain_prediction(record={}, shap_df=pd.DataFrame(), stream=False)
