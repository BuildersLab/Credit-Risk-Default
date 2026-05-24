"""
{{PROJECT_NAME}} model training.

Trains the champion model on processed features.
Run via: make train  or  python -m src.train

Owner(s):
"""

import os
import logging
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))

# TODO: tune these hyperparameters after running experiments in notebooks/03_modeling.ipynb
# Document the tuning rationale in docs/decisions.md
CHAMPION_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1,  # TODO: set to (negative_count / positive_count) for imbalanced data
    "eval_metric": "aucpr",
    "early_stopping_rounds": 30,
    "random_state": 42,
    "n_jobs": -1,
}


def load_data(processed_dir: Path = PROCESSED_DIR):
    X_train = pd.read_parquet(processed_dir / "X_train.parquet")
    X_val = pd.read_parquet(processed_dir / "X_val.parquet")
    y_train = pd.read_parquet(processed_dir / "y_train.parquet").squeeze()
    y_val = pd.read_parquet(processed_dir / "y_val.parquet").squeeze()
    logger.info("Loaded: X_train %s | X_val %s", X_train.shape, X_val.shape)
    return X_train, X_val, y_train, y_val


def train(X_train, y_train, X_val, y_val, params: dict = CHAMPION_PARAMS):
    """
    Train XGBoost with early stopping on validation PR-AUC.

    TODO: if this is not a binary classification problem, switch the model:
      - Multi-class: XGBClassifier with objective="multi:softprob"
      - Regression:  XGBRegressor with appropriate eval_metric
    Log your experiments in notebooks/03_modeling.ipynb before choosing the champion.
    """
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    val_probs = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, val_probs)
    logger.info("Validation {{PRIMARY_METRIC}}: %.4f", pr_auc)
    return model, pr_auc


def save_model(model, model_dir: Path = MODEL_DIR, name: str = "champion.pkl"):
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / name
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)
    return path


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()
    model, metric = train(X_train, y_train, X_val, y_val)
    save_model(model)
    logger.info("Training complete. {{PRIMARY_METRIC}}: %.4f", metric)
