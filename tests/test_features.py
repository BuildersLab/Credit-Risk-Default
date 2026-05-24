"""
Tests for src/features.py


TODO: update the fixture and assertions to match the actual features
built for Credit Risk Default.
"""

import pandas as pd
import numpy as np
import pytest
from src.features import build_all_features


@pytest.fixture
def sample_df():
    """
    TODO: update to match the cleaned/encoded output of data_pipeline for Credit Risk Default.
    """
    n = 50
    return pd.DataFrame({
        # TODO: add columns that features.py expects as input
        "date": pd.date_range("2020-06-01", periods=n, freq="3h"),
        "amount": np.random.uniform(10, 300, n),
        "entity_id": [str(i % 5) for i in range(n)],
        "category": np.random.randint(0, 5, n),
    })


def test_build_all_features_preserves_rows(sample_df):
    out = build_all_features(sample_df)
    assert len(out) == len(sample_df)


def test_build_all_features_adds_columns(sample_df):
    out = build_all_features(sample_df)
    assert len(out.columns) >= len(sample_df.columns)


def test_build_all_features_no_new_nulls(sample_df):
    out = build_all_features(sample_df)
    new_cols = [c for c in out.columns if c not in sample_df.columns]
    assert out[new_cols].isnull().sum().sum() == 0, \
        "Feature engineering introduced null values"


def test_build_all_features_no_inf(sample_df):
    out = build_all_features(sample_df)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    assert not np.isinf(out[numeric_cols].values).any(), \
        "Feature engineering introduced infinite values"
