"""Feature engineering utilities for fraud detection experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.config import (
    LEAKAGE_SUSPECT_FEATURES,
    REALTIME_SAFE_FEATURES,
    TARGET_COLUMN,
)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for both experimental feature sets."""
    result = df.copy()

    result["sender_balance_change"] = (
        result["newbalanceOrig"] - result["oldbalanceOrg"]
    )
    result["receiver_balance_change"] = (
        result["newbalanceDest"] - result["oldbalanceDest"]
    )

    result["amount_to_sender_old_balance"] = np.where(
        result["oldbalanceOrg"] > 0,
        result["amount"] / result["oldbalanceOrg"],
        0.0,
    )
    result["amount_to_receiver_old_balance"] = np.where(
        result["oldbalanceDest"] > 0,
        result["amount"] / result["oldbalanceDest"],
        0.0,
    )

    result["is_transfer_or_cash_out"] = result["type"].isin(
        ["TRANSFER", "CASH_OUT"]
    ).astype(int)

    result["sender_balance_before_zero"] = (
        result["oldbalanceOrg"] == 0
    ).astype(int)
    result["receiver_balance_before_zero"] = (
        result["oldbalanceDest"] == 0
    ).astype(int)

    return result


def get_feature_matrix(
    df: pd.DataFrame,
    feature_set: str = "realtime_safe",
) -> pd.DataFrame:
    """Return a feature matrix for the requested experiment.

    feature_set:
        - "realtime_safe": features available before transaction completion.
        - "leakage_suspect": includes post-transaction balances for diagnostic
          comparison only.
    """
    result = add_derived_features(df)

    if feature_set == "realtime_safe":
        columns = REALTIME_SAFE_FEATURES
    elif feature_set == "leakage_suspect":
        columns = LEAKAGE_SUSPECT_FEATURES
    else:
        raise ValueError(
            "feature_set must be 'realtime_safe' or 'leakage_suspect'."
        )

    missing_columns = set(columns) - set(result.columns)
    if missing_columns:
        raise ValueError(f"Missing feature columns: {sorted(missing_columns)}")

    return result[columns]


def get_target(df: pd.DataFrame) -> pd.Series:
    """Return the binary fraud target."""
    return df[TARGET_COLUMN]