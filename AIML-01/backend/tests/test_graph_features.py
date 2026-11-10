"""Tests for time-aware graph feature generation."""

import pandas as pd
from graph.graph_features import create_graph_features


def test_graph_features_do_not_use_same_step_transactions():
    dataframe = pd.DataFrame(
        {
            "step": [1, 1, 2],
            "amount": [100.0, 200.0, 300.0],
            "nameOrig": ["A", "A", "A"],
            "nameDest": ["B", "C", "B"],
        }
    )

    features = create_graph_features(dataframe)

    assert features.loc[0, "sender_prior_out_degree"] == 0
    assert features.loc[1, "sender_prior_out_degree"] == 0
    assert features.loc[0, "sender_receiver_prior_transaction_count"] == 0
    assert features.loc[1, "sender_receiver_prior_transaction_count"] == 0

    assert features.loc[2, "sender_prior_out_degree"] == 2
    assert features.loc[2, "sender_receiver_prior_transaction_count"] == 1
    assert features.loc[2, "sender_receiver_prior_total_amount"] == 100.0


def test_graph_features_use_only_prior_steps():
    dataframe = pd.DataFrame(
        {
            "step": [2, 1],
            "amount": [500.0, 100.0],
            "nameOrig": ["A", "A"],
            "nameDest": ["B", "B"],
        }
    )

    features = create_graph_features(dataframe)

    assert features.loc[1, "sender_receiver_prior_transaction_count"] == 0
    assert features.loc[0, "sender_receiver_prior_transaction_count"] == 1
    assert features.loc[0, "sender_receiver_prior_total_amount"] == 100.0