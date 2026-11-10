"""Data loading, validation, and temporal splitting utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_detection.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TIME_COLUMN,
)


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the PaySim transaction dataset."""
    df = pd.read_csv(path)
    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate required columns and basic data quality."""
    required_columns = {
        TARGET_COLUMN,
        TIME_COLUMN,
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")


def create_temporal_splits(
    df: pd.DataFrame,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronological train, validation, and test splits.

    The split boundaries are derived from the observed simulation steps,
    so the model is trained on earlier transactions and evaluated on later
    transactions.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave room for test data.")

    unique_steps = sorted(df[TIME_COLUMN].unique())
    train_end_index = int(len(unique_steps) * train_fraction)
    validation_end_index = int(len(unique_steps) * (train_fraction + validation_fraction))

    train_end_step = unique_steps[train_end_index - 1]
    validation_end_step = unique_steps[validation_end_index - 1]

    train_df = df[df[TIME_COLUMN] <= train_end_step].copy()
    validation_df = df[
        (df[TIME_COLUMN] > train_end_step) & (df[TIME_COLUMN] <= validation_end_step)
    ].copy()
    test_df = df[df[TIME_COLUMN] > validation_end_step].copy()

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError("One or more temporal splits are empty.")

    return train_df, validation_df, test_df