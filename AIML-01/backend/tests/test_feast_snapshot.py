"""Tests for the locally generated Feast feature snapshot."""

from pathlib import Path

import pandas as pd
from feature_repo.create_feature_data import create_feature_snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "feature_repo"
    / "data"
    / "transaction_features.parquet"
)

EXPECTED_COLUMNS = {
    "transaction_id",
    "event_timestamp",
    "amount",
    "oldbalance_org",
    "oldbalance_dest",
    "is_transfer_or_cash_out",
    "sender_balance_before_zero",
    "receiver_balance_before_zero",
}


def test_create_feature_snapshot(tmp_path, monkeypatch):
    """Feature snapshot should contain valid realtime-safe fields."""
    output_path = tmp_path / "transaction_features.parquet"

    monkeypatch.setattr(
        "feature_repo.create_feature_data.OUTPUT_PATH",
        output_path,
    )

    snapshot = create_feature_snapshot(row_limit=10)

    assert output_path.exists()
    assert len(snapshot) == 10
    assert set(snapshot.columns) == EXPECTED_COLUMNS
    assert snapshot["transaction_id"].is_unique
    assert pd.api.types.is_datetime64_any_dtype(snapshot["event_timestamp"])
    assert set(snapshot["is_transfer_or_cash_out"].unique()).issubset({0, 1})