"""Inspect fraud distribution and graph signal coverage in a PaySim window."""

from __future__ import annotations

import pandas as pd
from fraud_detection.config import RAW_DATA_PATH

from graph.graph_features import create_graph_features

COLUMNS = [
    "step",
    "amount",
    "nameOrig",
    "nameDest",
    "isFraud",
]


def main() -> None:
    """Inspect a chronological graph-analysis window."""
    dataframe = pd.read_csv(RAW_DATA_PATH, usecols=COLUMNS).head(200000)

    graph_input = dataframe.drop(columns=["isFraud"])
    graph_features = create_graph_features(graph_input)

    combined = pd.concat(
        [
            dataframe[["step", "isFraud"]].reset_index(drop=True),
            graph_features.reset_index(drop=True),
        ],
        axis=1,
    )

    fraud_by_step = (
        combined.groupby("step")["isFraud"]
        .agg(transaction_count="count", fraud_count="sum")
        .assign(fraud_rate=lambda frame: frame["fraud_count"] / frame["transaction_count"])
    )

    signal_columns = list(graph_features.columns)
    signal_coverage = (
        graph_features[signal_columns] > 0
    ).sum().rename("nonzero_rows")

    fraud_signal_summary = (
        combined.assign(
            has_graph_signal=combined[signal_columns].sum(axis=1) > 0
        )
        .groupby("has_graph_signal")["isFraud"]
        .agg(transaction_count="count", fraud_count="sum", fraud_rate="mean")
    )

    print("Window rows:", len(combined))
    print("Window fraud rows:", int(combined["isFraud"].sum()))
    print("\nFraud by step:")
    print(fraud_by_step)
    print("\nNonzero graph feature coverage:")
    print(signal_coverage)
    print("\nFraud by graph signal availability:")
    print(fraud_signal_summary)


if __name__ == "__main__":
    main()