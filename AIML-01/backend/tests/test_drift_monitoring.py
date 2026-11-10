"""Tests for drift-trigger policy and feature preparation."""

import pandas as pd
from monitoring.drift_report import (
    DRIFT_FEATURE_FRACTION_THRESHOLD,
    apply_controlled_drift,
    prepare_features,
    should_request_retraining_review,
)


def test_prepare_features_returns_expected_columns():
    dataframe = pd.DataFrame(
        {
            "type": ["TRANSFER", "PAYMENT"],
            "amount": [100.0, 50.0],
            "oldbalanceOrg": [100.0, 0.0],
            "oldbalanceDest": [0.0, 200.0],
        }
    )

    features = prepare_features(dataframe)

    assert list(features.columns) == [
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "type",
        "is_transfer_or_cash_out",
        "sender_balance_before_zero",
        "receiver_balance_before_zero",
    ]
    assert features["is_transfer_or_cash_out"].tolist() == [1, 0]
    assert features["sender_balance_before_zero"].tolist() == [0, 1]


def test_controlled_drift_multiplies_amount_only():
    dataframe = pd.DataFrame(
        {
            "amount": [10.0, 20.0],
            "type": ["TRANSFER", "PAYMENT"],
        }
    )

    drifted = apply_controlled_drift(dataframe)

    assert drifted["amount"].tolist() == [50.0, 100.0]
    assert drifted["type"].tolist() == ["TRANSFER", "PAYMENT"]


def test_retraining_review_policy_triggers_for_amount_drift():
    summary = {
        "drifted_columns": ["amount"],
        "drifted_feature_fraction": 1 / 7,
    }

    assert should_request_retraining_review(summary)


def test_retraining_review_policy_triggers_for_feature_fraction():
    summary = {
        "drifted_columns": ["oldbalanceOrg", "type", "oldbalanceDest"],
        "drifted_feature_fraction": DRIFT_FEATURE_FRACTION_THRESHOLD,
    }

    assert should_request_retraining_review(summary)


def test_retraining_review_policy_does_not_trigger_below_threshold():
    summary = {
        "drifted_columns": ["type"],
        "drifted_feature_fraction": 1 / 7,
    }

    assert not should_request_retraining_review(summary)