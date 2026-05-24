"""
{{PROJECT_NAME}} feature engineering.

Each function takes a DataFrame and returns a DataFrame with new columns added.
Functions never drop existing columns — dropping happens in data_pipeline.py.

Document each feature with:
  1. What it computes
  2. Why it is expected to be predictive (domain rationale)
  3. Any leakage risk to watch for

Owner(s):
"""

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based signals from the transaction/event timestamp.

    Domain rationale: TODO — explain why time matters for {{PROJECT_NAME}}.

    TODO: update "date_col" to the actual datetime column name.
    """
    df = df.copy()

    # TODO: replace "date_col" with the actual column
    # ts = df["date_col"]
    # df["hour"]       = ts.dt.hour
    # df["day_of_week"] = ts.dt.dayofweek
    # df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # df["month"]      = ts.dt.month

    return df


def add_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling or grouped aggregation features per entity (e.g. per customer, per card).

    Domain rationale: TODO — explain what behavioural pattern these capture.

    Leakage warning: if using rolling windows, ensure only PAST rows are used.
    Sort by the time column before computing and use expanding or shift(1).

    TODO: implement for {{PROJECT_NAME}}.
    """
    df = df.copy()

    # TODO: example rolling count per entity
    # df = df.sort_values("date_col")
    # df["txn_count_7d"] = (
    #     df.groupby("entity_id")["date_col"]
    #     .transform(lambda x: x.expanding().count() - 1)
    #     .clip(lower=0)
    #     .astype(int)
    # )

    return df


def add_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deviation of a value from the entity's historical mean.

    Domain rationale: TODO — explain what anomaly this captures.

    TODO: implement for {{PROJECT_NAME}}.
    """
    df = df.copy()

    # TODO: example amount deviation
    # entity_mean = df.groupby("entity_id")["amount"].transform("mean")
    # entity_std  = df.groupby("entity_id")["amount"].transform("std").fillna(1)
    # df["amount_z_score"] = (df["amount"] - entity_mean) / entity_std

    return df


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Project-specific features unique to {{PROJECT_NAME}}.

    TODO: add the features that are specific to this domain.
    These are the features that make this project distinct from others.
    Examples:
      - Fraud detection: haversine distance between cardholder and merchant
      - Credit risk:     debt-to-income ratio, derogatory mark count
      - Yield prediction: rolling 7-day rainfall, soil type encoding
    """
    df = df.copy()

    # TODO: add domain-specific features here

    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering functions in order.

    This is the single entrypoint called by data_pipeline.py and predict.py.
    Notebooks should call this function rather than individual helpers
    to guarantee consistency between training and inference.
    """
    df = add_temporal_features(df)
    df = add_aggregation_features(df)
    df = add_deviation_features(df)
    df = add_domain_features(df)
    return df


# TODO: update this list after features are implemented.
# This list must match exactly what the model was trained on.
# It is used by predict.py to select columns for inference.
FEATURE_COLS: list[str] = [
    # TODO: list all feature column names here after implementation
    # "hour", "day_of_week", "is_weekend", "amount_z_score", ...
]
