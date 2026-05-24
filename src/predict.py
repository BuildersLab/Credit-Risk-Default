"""
{{PROJECT_NAME}} inference module.

Used by the app to score new data in real time.

Owner(s):
"""

from __future__ import annotations
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

from src.features import build_all_features, FEATURE_COLS

load_dotenv()

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
THRESHOLD = float(os.getenv("OPERATING_THRESHOLD", "0.5"))


def load_model(model_dir: Path = MODEL_DIR, name: str = "champion.pkl"):
    """Load the champion model from disk."""
    return joblib.load(model_dir / name)


def score_tier(prob: float, threshold: float = THRESHOLD) -> str:
    """
    Map a probability to a human-readable score tier.

    TODO: adjust tier boundaries if needed for {{PROJECT_NAME}}.
    """
    if prob >= threshold * 1.5:
        return "High"
    elif prob >= threshold:
        return "Medium"
    return "Low"


def score(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Score a DataFrame of raw records.

    Applies feature engineering and returns the input DataFrame with
    three new columns: pred_prob, pred_flag, score_tier.
    """
    featured = build_all_features(df)
    available_cols = [c for c in FEATURE_COLS if c in featured.columns]
    X = featured[available_cols]
    probs = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["pred_prob"] = probs
    df["pred_flag"] = (probs >= THRESHOLD).astype(int)
    df["score_tier"] = [score_tier(p) for p in probs]
    return df


def score_batch(csv_path, model) -> pd.DataFrame:
    """
    Score a CSV file end-to-end. Used by the app's batch upload page.

    TODO: update datetime column parsing for {{PROJECT_NAME}}.
    """
    df = pd.read_csv(csv_path)
    # TODO: parse any datetime columns
    # df["date_col"] = pd.to_datetime(df["date_col"])
    return score(df, model)
