"""Train and track fraud-detection experiments with MLflow."""

from __future__ import annotations

import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from fraud_detection.config import (
    CATEGORICAL_FEATURES,
    MODEL_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    REPORT_DIR,
)
from fraud_detection.data import (
    create_temporal_splits,
    load_raw_data,
    validate_raw_data,
)
from fraud_detection.features import get_feature_matrix, get_target

SKOPS_TRUSTED_TYPES = [
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing for categorical and numeric features."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            )
        ],
        remainder="passthrough",
    )


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """Calculate XGBoost class weight from training labels only."""
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    return float(negative_count / positive_count)


def train_experiment(
    experiment_name: str,
    feature_set: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> tuple[Pipeline, float]:
    """Train one XGBoost experiment and log it to MLflow."""
    X_train = get_feature_matrix(train_df, feature_set=feature_set)
    y_train = get_target(train_df)

    X_validation = get_feature_matrix(validation_df, feature_set=feature_set)
    y_validation = get_target(validation_df)

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="aucpr",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )

    with mlflow.start_run(run_name=experiment_name):
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("subsample", 0.8)
        mlflow.log_param("colsample_bytree", 0.8)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("split_strategy", "temporal")

        model.fit(X_train, y_train)

        validation_scores = model.predict_proba(X_validation)[:, 1]
        validation_average_precision = average_precision_score(
            y_validation, validation_scores
        )

        mlflow.log_metric(
            "validation_average_precision", validation_average_precision
        )
        mlflow.log_metric("train_rows", len(train_df))
        mlflow.log_metric("validation_rows", len(validation_df))
        mlflow.log_metric("train_fraud_cases", int(y_train.sum()))
        mlflow.log_metric("validation_fraud_cases", int(y_validation.sum()))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    return model, validation_average_precision


def train_models() -> None:
    """Train diagnostic and realtime-safe XGBoost experiments."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    experiment = mlflow.get_experiment_by_name("fraud_detection_experiments")
    if experiment is None:
        mlflow.create_experiment(
            name="fraud_detection_experiments",
            artifact_location=(PROJECT_ROOT / "mlartifacts").resolve().as_uri(),
        )
    mlflow.set_experiment("fraud_detection_experiments")

    df = load_raw_data()
    validate_raw_data(df)

    train_df, validation_df, test_df = create_temporal_splits(df)

    leakage_model, leakage_validation_ap = train_experiment(
        experiment_name="leakage_suspect_xgboost",
        feature_set="leakage_suspect",
        train_df=train_df,
        validation_df=validation_df,
    )

    realtime_model, realtime_validation_ap = train_experiment(
        experiment_name="realtime_safe_xgboost",
        feature_set="realtime_safe",
        train_df=train_df,
        validation_df=validation_df,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(leakage_model, MODEL_DIR / "leakage_suspect_xgboost.joblib")
    joblib.dump(realtime_model, MODEL_DIR / "realtime_safe_xgboost.joblib")

    summary = {
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "leakage_suspect_validation_average_precision": float(
            leakage_validation_ap
        ),
        "realtime_safe_validation_average_precision": float(
            realtime_validation_ap
        ),
        "interpretation": (
            "The leakage-suspect experiment uses post-transaction balance "
            "fields and is retained only as a diagnostic upper bound. The "
            "realtime-safe experiment is the primary model for deployment."
        ),
    }

    with open(REPORT_DIR / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    train_models()