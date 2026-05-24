"""
Credit Risk Default data pipeline.

Loads raw data, cleans, encodes, and writes processed Parquet files.
Run via: make data  or  python -m src.data_pipeline

Owner(s):
"""

import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data/raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))


def load_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load raw data file(s) from data/raw/.

    TODO: replace with the actual filename(s) for Credit risk data - TBD.
    If multiple files need merging (e.g. transactions + identity),
    do the merge here and return a single DataFrame.
    """
    # TODO: update filename to match your dataset
    df = pd.read_csv(data_dir / "data.csv")
    logger.info("Loaded: %d rows, %d columns", *df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame.

    TODO: add project-specific cleaning steps:
    - Parse datetime columns
    - Drop columns with no analytical value (e.g. IDs, raw text)
    - Fix dtypes
    - Handle duplicates
    """
    df = df.copy()

    # TODO: parse datetime columns
    # df["date_col"] = pd.to_datetime(df["date_col"])

    # TODO: drop non-analytical columns
    # drop_cols = ["id", "name"]
    # df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    logger.info("Clean complete. Shape: %s", df.shape)
    return df


def encode(df: pd.DataFrame, fit_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Encode categorical columns.

    fit_df: if provided, encoding maps are computed from fit_df to prevent
    data leakage when encoding the test set. Always pass the training set here.

    TODO: add project-specific encoding:
    - Frequency encoding for high-cardinality categoricals
    - Label encoding for low-cardinality categoricals
    - Target encoding (with caution: use out-of-fold estimates)
    """
    df = df.copy()
    ref = fit_df if fit_df is not None else df

    # TODO: add encoding logic
    # Example frequency encoding:
    # freq_map = ref["merchant"].value_counts(normalize=True).to_dict()
    # df["merchant_freq"] = df["merchant"].map(freq_map).fillna(0)

    logger.info("Encode complete. Shape: %s", df.shape)
    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into train and validation sets.

    TODO: choose the right split strategy for this dataset:
    - Time-based split (recommended for time-series data — prevents leakage)
    - Stratified random split (for non-temporal data)
    Document the choice and rationale in docs/decisions.md.
    """
    # TODO: implement split
    # Time-based example:
    # cutoff = df["date_col"].max() - pd.DateOffset(months=2)
    # train = df[df["date_col"] <= cutoff]
    # val   = df[df["date_col"] >  cutoff]

    # Stratified random example:
    # from sklearn.model_selection import train_test_split
    # train, val = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)

    raise NotImplementedError("TODO: implement split() for Credit Risk Default")


def run_pipeline(
    data_dir: Path = DATA_DIR,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """
    Full end-to-end pipeline. Called by `make data`.

    Reads raw data, cleans, encodes, splits, and writes Parquet files.
    Running this twice should produce identical output (reproducible).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(data_dir)
    cleaned = clean(raw)
    encoded = encode(cleaned)
    train, val = split(encoded)

    # TODO: update TARGET_COL to the actual target column name
    TARGET_COL = "target"
    feature_cols = [c for c in train.columns if c != TARGET_COL]

    train[feature_cols].to_parquet(output_dir / "X_train.parquet", index=False)
    val[feature_cols].to_parquet(output_dir / "X_val.parquet", index=False)
    train[[TARGET_COL]].to_parquet(output_dir / "y_train.parquet", index=False)
    val[[TARGET_COL]].to_parquet(output_dir / "y_val.parquet", index=False)

    logger.info("Pipeline complete. Files written to %s", output_dir)


if __name__ == "__main__":
    run_pipeline()
