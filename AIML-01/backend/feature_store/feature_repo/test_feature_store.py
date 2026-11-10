"""Demonstrate and validate Feast historical and online feature retrieval."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from feast import FeatureStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_REPO_PATH = PROJECT_ROOT / "backend" / "feature_store" / "feature_repo"

FEATURE_REFERENCES = [
    "transaction_features:amount",
    "transaction_features:oldbalance_org",
    "transaction_features:oldbalance_dest",
    "transaction_features:is_transfer_or_cash_out",
    "transaction_features:sender_balance_before_zero",
    "transaction_features:receiver_balance_before_zero",
]

FEATURE_COLUMNS = [
    "amount",
    "oldbalance_org",
    "oldbalance_dest",
    "is_transfer_or_cash_out",
    "sender_balance_before_zero",
    "receiver_balance_before_zero",
]


def main() -> None:
    """Retrieve and compare the same registered features historically and online."""
    store = FeatureStore(repo_path=str(FEATURE_REPO_PATH))

    entity_ids = [
        "paysim_snapshot_0",
        "paysim_snapshot_1",
        "paysim_snapshot_2",
    ]

    entity_dataframe = pd.DataFrame(
        {
            "transaction_id": entity_ids,
            "event_timestamp": pd.to_datetime(
                [
                    "2026-01-01T01:00:00+00:00",
                    "2026-01-01T01:00:00+00:00",
                    "2026-01-01T01:00:00+00:00",
                ],
                utc=True,
            ),
        }
    )

    historical = store.get_historical_features(
        entity_df=entity_dataframe,
        features=FEATURE_REFERENCES,
    ).to_df()

    online = store.get_online_features(
        features=FEATURE_REFERENCES,
        entity_rows=[
            {"transaction_id": transaction_id}
            for transaction_id in entity_ids
        ],
    ).to_df()

    online = online[["transaction_id"] + FEATURE_COLUMNS]

    historical_sorted = historical.sort_values("transaction_id").reset_index(
        drop=True
    )
    online_sorted = online.sort_values("transaction_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(
        historical_sorted[["transaction_id"] + FEATURE_COLUMNS],
        online_sorted,
        check_dtype=False,
        check_like=False,
    )

    print("Historical retrieval:")
    print(historical_sorted)

    print("\nOnline retrieval:")
    print(online_sorted)

    print(
        "\nTraining-serving consistency check passed: "
        "historical and online feature values match."
    )


if __name__ == "__main__":
    main()