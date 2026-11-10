"""Model loading and prediction logic for the scoring API."""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd
from fraud_detection.config import MODEL_DIR, REPORT_DIR
from fraud_detection.features import get_feature_matrix

PRIMARY_MODEL_PATH = MODEL_DIR / "realtime_safe_xgboost.joblib"
THRESHOLD_REPORT_PATH = REPORT_DIR / "evaluation_results.json"


@lru_cache(maxsize=1)
def load_model():
    """Load the trained realtime-safe XGBoost pipeline."""
    if not PRIMARY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Primary model not found: {PRIMARY_MODEL_PATH}. "
            "Run `python -m fraud_detection.train` first."
        )
    return joblib.load(PRIMARY_MODEL_PATH)


@lru_cache(maxsize=1)
def load_threshold() -> float:
    """Load the validation-selected decision threshold."""
    if not THRESHOLD_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation report not found: {THRESHOLD_REPORT_PATH}. "
            "Run `python -m fraud_detection.evaluate` first."
        )

    with open(THRESHOLD_REPORT_PATH, "r", encoding="utf-8") as file:
        results = json.load(file)

    return float(results["realtime_safe"]["threshold"])


def score_transaction(transaction: dict) -> dict:
    """Score one transaction and return an analyst-friendly response."""
    model = load_model()
    threshold = load_threshold()

    transaction_df = pd.DataFrame([transaction])
    features = get_feature_matrix(transaction_df, feature_set="realtime_safe")

    fraud_probability = float(model.predict_proba(features)[:, 1][0])
    is_alert = fraud_probability >= threshold

    if fraud_probability >= 0.90:
        recommended_action = "Block for immediate review"
    elif is_alert:
        recommended_action = "Send to analyst review queue"
    else:
        recommended_action = "Allow transaction"

    return {
        "transaction_id": transaction["transaction_id"],
        "fraud_probability": fraud_probability,
        "risk_score": round(fraud_probability * 100, 2),
        "is_alert": is_alert,
        "model_name": "realtime_safe_xgboost",
        "threshold": threshold,
        "recommended_action": recommended_action,
    }