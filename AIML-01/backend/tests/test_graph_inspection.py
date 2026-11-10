"""Tests for graph-analysis interpretation helpers."""

import pandas as pd


def calculate_fraud_rate_by_signal(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize fraud prevalence by graph-signal availability."""
    return (
        dataframe.groupby("has_graph_signal")["isFraud"]
        .agg(transaction_count="count", fraud_count="sum", fraud_rate="mean")
    )


def test_fraud_rate_summary_by_signal():
    dataframe = pd.DataFrame(
        {
            "has_graph_signal": [False, False, True, True, True],
            "isFraud": [1, 0, 0, 0, 0],
        }
    )

    summary = calculate_fraud_rate_by_signal(dataframe)

    assert summary.loc[False, "transaction_count"] == 2
    assert summary.loc[False, "fraud_count"] == 1
    assert summary.loc[False, "fraud_rate"] == 0.5
    assert summary.loc[True, "transaction_count"] == 3
    assert summary.loc[True, "fraud_count"] == 0
    assert summary.loc[True, "fraud_rate"] == 0.0