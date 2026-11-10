"""Create a patient-level train/validation split for NIH ChestX-ray14.

The official NIH test list remains the held-out test set. This module assigns
patients, rather than individual images, from the train/validation pool to
either train or validation.

The output manifest remains outside Git because it contains patient-level
metadata.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REQUIRED_COLUMNS = {"image_index", "image_path", "patient_id", "split"}
OUTPUT_COLUMNS = [
    "image_index",
    "image_path",
    "patient_id",
    "follow_up_number",
    "view_position",
    "split",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read and validate the base NIH manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest contains no rows: {path}")

    return rows


def assign_patient_splits(
    rows: list[dict[str, str]],
    validation_fraction: float,
    random_seed: int,
) -> dict[str, str]:
    """Assign each train/validation patient to train or validation."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    patient_rows: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        if row["split"] == "test":
            continue
        if row["split"] != "train_val":
            raise ValueError(f"Unexpected base split value: {row['split']!r}")
        patient_rows[row["patient_id"]].append(row)

    patient_ids = sorted(patient_rows)
    rng = random.Random(random_seed)
    rng.shuffle(patient_ids)

    validation_count = round(len(patient_ids) * validation_fraction)
    validation_ids = set(patient_ids[:validation_count])

    return {
        patient_id: "validation" if patient_id in validation_ids else "train"
        for patient_id in patient_ids
    }


def create_patient_split(
    manifest_path: Path,
    output_path: Path,
    validation_fraction: float,
    random_seed: int,
) -> dict[str, int]:
    """Create the final patient-level split manifest."""
    rows = read_manifest(manifest_path)
    patient_splits = assign_patient_splits(
        rows,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
    )

    output_rows: list[dict[str, str]] = []
    counts = {"train": 0, "validation": 0, "test": 0}
    patient_split_counts: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        final_split = (
            "test"
            if row["split"] == "test"
            else patient_splits[row["patient_id"]]
        )
        counts[final_split] += 1
        patient_split_counts[row["patient_id"]].add(final_split)

        output_row = dict(row)
        output_row["split"] = final_split
        output_rows.append(output_row)

    leaked_patients = [
        patient_id
        for patient_id, splits in patient_split_counts.items()
        if len(splits) > 1
    ]
    if leaked_patients:
        raise ValueError(
            f"Patient leakage detected for {len(leaked_patients)} patients."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=OUTPUT_COLUMNS, extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(output_rows)

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/classification/config.yaml"),
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset = config["dataset"]
    split_config = config["split"]

    manifest_path = Path(dataset["manifest_file"])
    output_path = Path(dataset["manifest_file"]).with_name(
        "nih_chestxray14_patient_splits.csv"
    )

    try:
        counts = create_patient_split(
            manifest_path=manifest_path,
            output_path=output_path,
            validation_fraction=float(split_config["validation_fraction"]),
            random_seed=int(split_config["random_seed"]),
        )
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(f"PASS: patient-level split written to {output_path}")
    print(f"Train images: {counts['train']}")
    print(f"Validation images: {counts['validation']}")
    print(f"Held-out test images: {counts['test']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())