
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

FILES = {
    "train": DATA_DIR / "train.csv.gz",
    "validation": DATA_DIR / "validation.csv.gz",
    "test": DATA_DIR / "test.csv.gz",
}


def load_split(name, path):
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")

    return pd.read_csv(
        path,
        usecols=["userId", "movieId", "timestamp"],
    )


def main():
    print("Loading split files...")

    train = load_split("train", FILES["train"])
    validation = load_split("validation", FILES["validation"])
    test = load_split("test", FILES["test"])

    print("\n--- ROW COUNTS ---")
    print(f"Train:      {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test:       {len(test):,}")

    train_max = train.groupby("userId")["timestamp"].max()
    validation_min = validation.groupby("userId")["timestamp"].min()
    validation_max = validation.groupby("userId")["timestamp"].max()
    test_min = test.groupby("userId")["timestamp"].min()

    common_train_validation = train_max.index.intersection(
        validation_min.index
    )
    common_validation_test = validation_max.index.intersection(
        test_min.index
    )

    train_before_validation = (
        train_max.loc[common_train_validation]
        <= validation_min.loc[common_train_validation]
    ).all()

    validation_before_test = (
        validation_max.loc[common_validation_test]
        <= test_min.loc[common_validation_test]
    ).all()

    print("\n--- CHRONOLOGICAL CHECKS ---")
    print(
        "Train ends before validation begins per user: "
        f"{train_before_validation}"
    )
    print(
        "Validation ends before test begins per user: "
        f"{validation_before_test}"
    )

    print("\n--- DUPLICATE INTERACTIONS ---")
    for name, df in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        duplicates = df.duplicated(["userId", "movieId"]).sum()
        print(f"{name}: {duplicates:,}")

    if not train_before_validation or not validation_before_test:
        raise ValueError(
            "Chronological validation failed. Inspect timestamp ties "
            "and split boundaries before proceeding."
        )

    print("\nSplit validation checks passed.")


if __name__ == "__main__":
    main()
