"""
{{PROJECT_NAME}} SHAP explainability helpers.

Used by notebooks (analysis) and the app (per-prediction explanations).
Keeping SHAP logic here ensures training and inference use identical code.

"""

from __future__ import annotations
import numpy as np
import pandas as pd
import shap


def get_explainer(model, X_background: pd.DataFrame) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer. X_background: 200 to 500 training rows."""
    return shap.TreeExplainer(
        model,
        data=X_background,
        feature_perturbation="interventional",
    )


def get_shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for a DataFrame of predictions."""
    return explainer.shap_values(X)


def top_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    n: int = 10,
) -> pd.DataFrame:
    """Global feature importance from mean absolute SHAP values."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        .sort_values("importance", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def waterfall_data(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_values: np.ndarray,
    idx: int = 0,
    n: int = 8,
) -> pd.DataFrame:
    """Per-prediction SHAP data for a waterfall chart. Used by the app and gemini.py."""
    sv = shap_values[idx]
    fv = feature_values[idx]
    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv,
        "feature_value": fv,
    })
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).head(n).reset_index(drop=True)
    df["direction"] = df["shap_value"].apply(
        lambda x: "increases prediction" if x > 0 else "decreases prediction"
    )
    return df
