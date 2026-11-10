"""Create a local Feast feature snapshot from PaySim transactions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "backend"
    / "feature_store"
    / "feature_repo"
    / "data"
    / "transaction_features.parquet"
)

SIMULATION_START = pd.Timestamp(
    datetime(2026, 1, 1, tzinfo=timezone.utc)
)

COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]


def create_feature_snapshot(row_limit: int = 50000) -> pd.DataFrame:
    """Create realtime-safe Feast features from a chronological PaySim sample."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"PaySim CSV not found: {RAW_DATA_PATH}")

    dataframe = pd.read_csv(
        RAW_DATA_PATH,
        usecols=COLUMNS,
        nrows=row_limit,
    ).copy()

    dataframe["transaction_id"] = [
        f"paysim_snapshot_{index}"
        for index in dataframe.index
    ]

    dataframe["event_timestamp"] = (
        SIMULATION_START
        + pd.to_timedelta(dataframe["step"], unit="h")
    )

    dataframe["oldbalance_org"] = dataframe["oldbalanceOrg"].astype("float32")
    dataframe["oldbalance_dest"] = dataframe["oldbalanceDest"].astype("float32")
    dataframe["amount"] = dataframe["amount"].astype("float32")
    dataframe["is_transfer_or_cash_out"] = (
        dataframe["type"].isin(["TRANSFER", "CASH_OUT"]).astype("int64")
    )
    dataframe["sender_balance_before_zero"] = (
        dataframe["oldbalanceOrg"] == 0
    ).astype("int64")
    dataframe["receiver_balance_before_zero"] = (
        dataframe["oldbalanceDest"] == 0
    ).astype("int64")

    output_columns = [
        "transaction_id",
        "event_timestamp",
        "amount",
        "oldbalance_org",
        "oldbalance_dest",
        "is_transfer_or_cash_out",
        "sender_balance_before_zero",
        "receiver_balance_before_zero",
    ]

    snapshot = dataframe[output_columns].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(OUTPUT_PATH, index=False)

    return snapshot


if __name__ == "__main__":
    result = create_feature_snapshot()
    print(f"Created feature snapshot: {OUTPUT_PATH}")
    print(f"Rows: {len(result)}")
    print(result.head())