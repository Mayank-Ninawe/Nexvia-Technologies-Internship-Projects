"""Evaluate leakage-suspect and realtime-safe fraud models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from fraud_detection.config import MODEL_DIR, REPORT_DIR, TIME_COLUMN
from fraud_detection.data import (
    create_temporal_splits,
    load_raw_data,
    validate_raw_data,
)
from fraud_detection.features import get_feature_matrix, get_target


def select_threshold(
    y_true: pd.Series,
    scores: np.ndarray,
    minimum_precision: float = 0.50,
) -> float:
    """Select a validation threshold using a precision constraint."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    candidate_indices = np.where(precision[:-1] >= minimum_precision)[0]

    if len(candidate_indices) > 0:
        best_index = candidate_indices[np.argmax(recall[candidate_indices])]
        return float(thresholds[best_index])

    f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    best_index = int(np.argmax(f1_scores))
    return float(thresholds[best_index])


def evaluate_model(
    model_path: Path,
    model_name: str,
    feature_set: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate a saved model on the untouched test set."""
    model = joblib.load(model_path)
    scores = model.predict_proba(X_test)[:, 1]
    predictions = (scores >= threshold).astype(int)

    matrix = confusion_matrix(y_test, predictions)

    metrics = {
        "model": model_name,
        "feature_set": feature_set,
        "threshold": threshold,
        "test_average_precision": float(average_precision_score(y_test, scores)),
        "test_precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "true_negatives": int(matrix[0, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_positives": int(matrix[1, 1]),
    }
    metrics.update(
        {
            "f2_score": float(fbeta_score(y_test, predictions, beta=2, zero_division=0)),
            "alert_rate": float(predictions.mean()),
            "false_positive_rate": float(
                matrix[0, 1] / max(matrix[0, 0] + matrix[0, 1], 1)
            ),
            "brier_score": float(brier_score_loss(y_test, scores)),
        }
    )

    return metrics, scores, matrix


def evaluate_temporal_windows(
    model,
    test_df: pd.DataFrame,
    feature_set: str,
    threshold: float,
    window_count: int = 3,
) -> list[dict]:
    """Measure fixed-model stability across chronological test windows."""
    steps = sorted(test_df[TIME_COLUMN].unique())
    windows = np.array_split(np.asarray(steps), window_count)
    results = []

    for index, window_steps in enumerate(windows, start=1):
        if len(window_steps) == 0:
            continue
        window_df = test_df[test_df[TIME_COLUMN].isin(window_steps)]
        features = get_feature_matrix(window_df, feature_set=feature_set)
        targets = get_target(window_df)
        scores = model.predict_proba(features)[:, 1]
        predictions = (scores >= threshold).astype(int)
        matrix = confusion_matrix(targets, predictions, labels=[0, 1])
        results.append(
            {
                "window": index,
                "start_step": int(window_steps[0]),
                "end_step": int(window_steps[-1]),
                "rows": len(window_df),
                "fraud_rows": int(targets.sum()),
                "average_precision": float(average_precision_score(targets, scores)),
                "precision": float(
                    precision_score(targets, predictions, zero_division=0)
                ),
                "recall": float(recall_score(targets, predictions, zero_division=0)),
                "f2_score": float(
                    fbeta_score(targets, predictions, beta=2, zero_division=0)
                ),
                "alert_rate": float(predictions.mean()),
                "false_positive_rate": float(
                    matrix[0, 1] / max(matrix[0, 0] + matrix[0, 1], 1)
                ),
                "brier_score": float(brier_score_loss(targets, scores)),
            }
        )
    return results


def run_evaluation() -> None:
    """Evaluate both models on the untouched test set."""
    df = load_raw_data()
    validate_raw_data(df)

    _, validation_df, test_df = create_temporal_splits(df)

    X_validation_safe = get_feature_matrix(
        validation_df, feature_set="realtime_safe"
    )
    X_test_safe = get_feature_matrix(test_df, feature_set="realtime_safe")

    X_validation_leakage = get_feature_matrix(
        validation_df, feature_set="leakage_suspect"
    )
    X_test_leakage = get_feature_matrix(test_df, feature_set="leakage_suspect")

    y_validation = get_target(validation_df)
    y_test = get_target(test_df)

    realtime_model = joblib.load(MODEL_DIR / "realtime_safe_xgboost.joblib")
    leakage_model = joblib.load(MODEL_DIR / "leakage_suspect_xgboost.joblib")

    realtime_validation_scores = realtime_model.predict_proba(
        X_validation_safe
    )[:, 1]
    leakage_validation_scores = leakage_model.predict_proba(
        X_validation_leakage
    )[:, 1]

    realtime_threshold = select_threshold(y_validation, realtime_validation_scores)
    leakage_threshold = select_threshold(y_validation, leakage_validation_scores)

    realtime_metrics, realtime_scores, realtime_matrix = evaluate_model(
        MODEL_DIR / "realtime_safe_xgboost.joblib",
        "realtime_safe_xgboost",
        "realtime_safe",
        X_test_safe,
        y_test,
        realtime_threshold,
    )
    temporal_windows = evaluate_temporal_windows(
        realtime_model,
        test_df,
        "realtime_safe",
        realtime_threshold,
    )

    leakage_metrics, _, _ = evaluate_model(
        MODEL_DIR / "leakage_suspect_xgboost.joblib",
        "leakage_suspect_xgboost",
        "leakage_suspect",
        X_test_leakage,
        y_test,
        leakage_threshold,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_DIR / "evaluation_results.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "realtime_safe": realtime_metrics,
                "leakage_suspect": leakage_metrics,
                "realtime_safe_temporal_windows": temporal_windows,
            },
            file,
            indent=2,
        )

    precision, recall, _ = precision_recall_curve(y_test, realtime_scores)

    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, label="Realtime-safe XGBoost")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Realtime-safe Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "realtime_safe_precision_recall_curve.png", dpi=150)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        realtime_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Legitimate", "Predicted Fraud"],
        yticklabels=["Actual Legitimate", "Actual Fraud"],
    )
    plt.title("Realtime-safe XGBoost Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "realtime_safe_confusion_matrix.png", dpi=150)

    print(
        json.dumps(
            {
                "realtime_safe": realtime_metrics,
                "leakage_suspect": leakage_metrics,
                "realtime_safe_temporal_windows": temporal_windows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run_evaluation()