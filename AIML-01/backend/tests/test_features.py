"""Tests for feature engineering."""

import pandas as pd
from fraud_detection.features import add_derived_features, get_feature_matrix


def test_add_derived_features():
    df = pd.DataFrame(
        {
            "type": ["TRANSFER"],
            "amount": [100.0],
            "oldbalanceOrg": [500.0],
            "newbalanceOrig": [400.0],
            "oldbalanceDest": [200.0],
            "newbalanceDest": [300.0],
        }
    )

    result = add_derived_features(df)

    assert result["sender_balance_change"].iloc[0] == -100.0
    assert result["receiver_balance_change"].iloc[0] == 100.0
    assert result["amount_to_sender_old_balance"].iloc[0] == 0.2
    assert result["is_transfer_or_cash_out"].iloc[0] == 1
    assert result["sender_balance_before_zero"].iloc[0] == 0
    assert result["receiver_balance_before_zero"].iloc[0] == 0


def test_get_realtime_safe_feature_matrix_columns():
    df = pd.DataFrame(
        {
            "type": ["PAYMENT"],
            "amount": [50.0],
            "oldbalanceOrg": [0.0],
            "newbalanceOrig": [0.0],
            "oldbalanceDest": [0.0],
            "newbalanceDest": [0.0],
        }
    )

    features = get_feature_matrix(df, feature_set="realtime_safe")

    expected_columns = {
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "amount_to_sender_old_balance",
        "amount_to_receiver_old_balance",
        "is_transfer_or_cash_out",
        "sender_balance_before_zero",
        "receiver_balance_before_zero",
    }

    assert set(features.columns) == expected_columns
    assert "sender_balance_change" not in features.columns
    assert len(features) == 1


def test_get_leakage_suspect_feature_matrix_columns():
    df = pd.DataFrame(
        {
            "type": ["TRANSFER"],
            "amount": [100.0],
            "oldbalanceOrg": [500.0],
            "newbalanceOrig": [400.0],
            "oldbalanceDest": [200.0],
            "newbalanceDest": [300.0],
        }
    )

    features = get_feature_matrix(df, feature_set="leakage_suspect")

    assert "sender_balance_change" in features.columns
    assert "receiver_balance_change" in features.columns
    assert len(features) == 1