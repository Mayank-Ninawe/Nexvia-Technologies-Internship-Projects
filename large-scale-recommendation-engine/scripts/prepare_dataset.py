
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ml-25m"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RATINGS_FILE = RAW_DIR / "ratings.csv"

# Per-user chronological split:
# first 80% -> train, next 10% -> validation, last 10% -> test.
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10

# Keep the minimum small enough to retain users with short histories.
MIN_RATINGS_PER_USER = 3


def main():
    if not RATINGS_FILE.exists():
        raise FileNotFoundError(
            f"Ratings file not found: {RATINGS_FILE}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MovieLens ratings...")

    ratings = pd.read_csv(
        RATINGS_FILE,
        usecols=["userId", "movieId", "rating", "timestamp"],
    )

    print(f"Loaded {len(ratings):,} ratings.")

    # Sort each user's interactions from oldest to newest.
    ratings = ratings.sort_values(
        ["userId", "timestamp", "movieId"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Users with too few ratings cannot provide reliable
    # train, validation, and test examples.
    user_counts = ratings.groupby("userId")["movieId"].transform("size")
    eligible = ratings[user_counts >= MIN_RATINGS_PER_USER].copy()

    excluded_users = ratings["userId"].nunique() - eligible["userId"].nunique()

    print(f"Eligible users: {eligible['userId'].nunique():,}")
    print(f"Excluded users with too few ratings: {excluded_users:,}")

    # Calculate each rating's chronological position within its user.
    eligible["user_position"] = (
        eligible.groupby("userId").cumcount()
    )

    eligible["user_history_size"] = (
        eligible.groupby("userId")["movieId"].transform("size")
    )

    eligible["position_fraction"] = (
        eligible["user_position"] / eligible["user_history_size"]
    )

    train = eligible[
        eligible["position_fraction"] < TRAIN_RATIO
    ].copy()

    validation = eligible[
        (eligible["position_fraction"] >= TRAIN_RATIO)
        & (
            eligible["position_fraction"]
            < TRAIN_RATIO + VALIDATION_RATIO
        )
    ].copy()

    test = eligible[
        eligible["position_fraction"] >= TRAIN_RATIO + VALIDATION_RATIO
    ].copy()

    # Remove helper columns from the saved datasets.
    output_columns = ["userId", "movieId", "rating", "timestamp"]

    train = train[output_columns]
    validation = validation[output_columns]
    test = test[output_columns]

    # Save as compressed CSV to reduce disk usage.
    train_path = PROCESSED_DIR / "train.csv.gz"
    validation_path = PROCESSED_DIR / "validation.csv.gz"
    test_path = PROCESSED_DIR / "test.csv.gz"

    train.to_csv(train_path, index=False, compression="gzip")
    validation.to_csv(validation_path, index=False, compression="gzip")
    test.to_csv(test_path, index=False, compression="gzip")

    print("\n--- SPLIT SUMMARY ---")
    print(f"Train:      {len(train):,} ratings ({len(train) / len(eligible):.1%})")
    print(
        f"Validation: {len(validation):,} ratings "
        f"({len(validation) / len(eligible):.1%})"
    )
    print(f"Test:       {len(test):,} ratings ({len(test) / len(eligible):.1%})")

    print("\n--- USER OVERLAP ---")
    print(
        "Users in all three splits: "
        f"{len(set(train['userId']) & set(validation['userId']) & set(test['userId'])):,}"
    )

    print("\n--- SAVED FILES ---")
    print(train_path)
    print(validation_path)
    print(test_path)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
