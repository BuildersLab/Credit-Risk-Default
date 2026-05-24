"""
Tests for src/data_pipeline.py


These tests verify that the pipeline produces correctly shaped, typed, and split
output. They do not test model accuracy. They run on every PR via CI.

TODO: update fixtures and assertions to match {{PROJECT_NAME}} data schema.
"""

import pandas as pd
import numpy as np
import pytest
from src.data_pipeline import clean, encode, split


@pytest.fixture
def sample_df():
    """
    Minimal synthetic DataFrame that mirrors the schema of {{DATASET_NAME}}.

    TODO: update columns to match the actual dataset for {{PROJECT_NAME}}.
    Keep this small (50 to 100 rows) — it runs in CI without real data files.
    """
    n = 100
    return pd.DataFrame({
        # TODO: add columns matching {{DATASET_NAME}} schema
        "id": range(n),
        "date": pd.date_range("2020-01-01", periods=n, freq="h"),
        "amount": np.random.uniform(10, 500, n),
        "category": [f"cat_{i % 4}" for i in range(n)],
        "target": ([0] * 95) + ([1] * 5),
    })


def test_clean_preserves_row_count(sample_df):
    cleaned = clean(sample_df)
    assert len(cleaned) == len(sample_df)


def test_clean_parses_datetime(sample_df):
    cleaned = clean(sample_df)
    # TODO: update "date" to the actual datetime column name
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])


def test_encode_adds_expected_columns(sample_df):
    cleaned = clean(sample_df)
    encoded = encode(cleaned)
    # TODO: update expected encoded column names for {{PROJECT_NAME}}
    # assert "category_freq" in encoded.columns
    assert len(encoded) == len(sample_df)


def test_encode_no_nan_introduced(sample_df):
    cleaned = clean(sample_df)
    encoded = encode(cleaned)
    assert not encoded.isnull().any().any()


def test_split_no_overlap(sample_df):
    cleaned = clean(sample_df)
    encoded = encode(cleaned)
    train, val = split(encoded)
    assert len(train) + len(val) == len(encoded)
    assert set(train.index).isdisjoint(set(val.index))


def test_split_train_larger_than_val(sample_df):
    cleaned = clean(sample_df)
    encoded = encode(cleaned)
    train, val = split(encoded)
    assert len(train) > len(val)
