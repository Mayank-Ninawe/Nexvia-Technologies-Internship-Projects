"""Time-aware graph features for PaySim transaction scoring."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from fraud_detection.config import RAW_DATA_PATH

REQUIRED_COLUMNS = [
    "step",
    "amount",
    "nameOrig",
    "nameDest",
]


@dataclass
class RelationshipState:
    """Historical aggregate state for one directed account relationship."""

    transaction_count: int = 0
    total_amount: float = 0.0


class HistoricalGraphFeatureBuilder:
    """Build graph-style features without using current/future transactions.

    For every transaction at step t, generated features reflect only graph
    state created by transactions with step strictly less than t.
    """

    def __init__(self) -> None:
        self.outgoing_neighbors: dict[str, set[str]] = defaultdict(set)
        self.incoming_neighbors: dict[str, set[str]] = defaultdict(set)
        self.outgoing_transaction_counts: dict[str, int] = defaultdict(int)
        self.incoming_transaction_counts: dict[str, int] = defaultdict(int)
        self.relationships: dict[tuple[str, str], RelationshipState] = defaultdict(
            RelationshipState
        )

    def _features_for_row(self, row: pd.Series) -> dict:
        """Generate historical graph features before updating graph state."""
        sender = row["nameOrig"]
        receiver = row["nameDest"]
        relationship = self.relationships[(sender, receiver)]

        return {
            "sender_prior_out_degree": len(self.outgoing_neighbors[sender]),
            "receiver_prior_in_degree": len(self.incoming_neighbors[receiver]),
            "sender_prior_outgoing_transaction_count": (
                self.outgoing_transaction_counts[sender]
            ),
            "receiver_prior_incoming_transaction_count": (
                self.incoming_transaction_counts[receiver]
            ),
            "sender_receiver_prior_transaction_count": (
                relationship.transaction_count
            ),
            "sender_receiver_prior_total_amount": relationship.total_amount,
        }

    def _update_state(self, row: pd.Series) -> None:
        """Update graph state after all features at a time step are generated."""
        sender = row["nameOrig"]
        receiver = row["nameDest"]
        amount = float(row["amount"])

        self.outgoing_neighbors[sender].add(receiver)
        self.incoming_neighbors[receiver].add(sender)
        self.outgoing_transaction_counts[sender] += 1
        self.incoming_transaction_counts[receiver] += 1

        relationship = self.relationships[(sender, receiver)]
        relationship.transaction_count += 1
        relationship.total_amount += amount

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return prior-only graph features for chronologically ordered data."""
        required_columns = {"step", "amount", "nameOrig", "nameDest"}
        missing_columns = required_columns - set(dataframe.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns required for graph features: "
                f"{sorted(missing_columns)}"
            )

        sorted_dataframe = dataframe.sort_values(
            ["step"],
            kind="stable",
        ).copy()

        feature_records: list[dict] = []

        for _, step_dataframe in sorted_dataframe.groupby("step", sort=True):
            step_features = [
                self._features_for_row(row)
                for _, row in step_dataframe.iterrows()
            ]
            feature_records.extend(step_features)

            for _, row in step_dataframe.iterrows():
                self._update_state(row)

        features = pd.DataFrame(feature_records, index=sorted_dataframe.index)
        return features.reindex(dataframe.index)


def load_graph_feature_window(
    max_rows: int = 200000,
    max_step: int | None = None,
) -> pd.DataFrame:
    """Load a chronological PaySim slice for graph feature generation."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"PaySim CSV not found at: {RAW_DATA_PATH}")

    dataframe = pd.read_csv(RAW_DATA_PATH, usecols=REQUIRED_COLUMNS)

    if max_step is not None:
        dataframe = dataframe[dataframe["step"] <= max_step].copy()

    return dataframe.head(max_rows).copy()


def create_graph_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create time-aware graph features for a dataframe."""
    builder = HistoricalGraphFeatureBuilder()
    return builder.transform(dataframe)


if __name__ == "__main__":
    data = load_graph_feature_window(max_rows=10000, max_step=24)
    graph_features = create_graph_features(data)

    print(graph_features.head())
    print(graph_features.describe())