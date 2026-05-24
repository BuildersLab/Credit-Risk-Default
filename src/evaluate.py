"""
Credit Risk Default model evaluation.

Produces metrics framed as business impact.
The summary sentence from this module is used in the final demo presentation.

Run via: make evaluate  or  python -m src.evaluate

Owner(s):
"""

import os
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, confusion_matrix

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
THRESHOLD = float(os.getenv("OPERATING_THRESHOLD", "0.5"))


def load_artifacts(processed_dir: Path, model_dir: Path):
    X_test = pd.read_parquet(processed_dir / "X_test.parquet")
    y_test = pd.read_parquet(processed_dir / "y_test.parquet").squeeze()
    model = joblib.load(model_dir / "champion.pkl")
    return X_test, y_test, model


def compute_business_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    value_col: pd.Series = None,
    threshold: float = THRESHOLD,
) -> dict:
    """
    Compute metrics framed as business impact.

    value_col: a Series of monetary or impact values per row (e.g. loan amount,
    transaction amount, crop yield). Used to compute dollar/unit impact.
    If None, falls back to count-based metrics only.

    TODO: update the summary_sentence template to match the Credit Risk Default framing.
    The sentence should answer: what does this model save or catch for NorthBay Bank TBD?
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    pr_auc = average_precision_score(y_true, y_prob)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fp_per_1000 = (fp / max((y_true == 0).sum(), 1)) * 1000

    # TODO: replace with Credit Risk Default-specific business framing
    # Example for fraud: dollar value of fraud caught
    # Example for credit risk: expected loss avoided
    # Example for yield: tonnes of crop yield predicted accurately
    if value_col is not None:
        total_value = value_col[y_true == 1].sum()
        caught_value = value_col[(y_true == 1) & (y_pred == 1)].sum()
        pct_value_caught = caught_value / total_value * 100 if total_value > 0 else 0
    else:
        caught_value = float(tp)
        pct_value_caught = recall * 100

    # TODO: update this sentence for Credit Risk Default
    summary_sentence = (
        f"At the {threshold:.0%} threshold, the model achieves "
        f"{recall:.1%} recall and {precision:.1%} precision, "
        f"generating {fp_per_1000:.1f} false alerts per 1,000 negative cases."
    )

    return {
        "pr_auc": pr_auc,
        "recall": recall,
        "precision": precision,
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "caught_value": caught_value,
        "pct_value_caught": pct_value_caught,
        "fp_per_1000": fp_per_1000,
        "summary_sentence": summary_sentence,
    }


def print_report(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Credit Risk Default MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"\nPR-AUC:              {metrics['pr_auc']:.4f}")
    print(f"Recall:                {metrics['recall']:.4f}")
    print(f"Precision:             {metrics['precision']:.4f}")
    print(f"Operating threshold:   {metrics['threshold']:.2f}")
    print(f"False alerts/1,000:    {metrics['fp_per_1000']:.2f}")
    print(f"\n{metrics['summary_sentence']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    X_test, y_test, model = load_artifacts(PROCESSED_DIR, MODEL_DIR)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_business_metrics(y_test, y_prob)
    print_report(metrics)
