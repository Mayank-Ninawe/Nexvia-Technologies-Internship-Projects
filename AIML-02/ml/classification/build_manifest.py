"""Build a validated manifest for the NIH ChestX-ray14 dataset.

The official NIH metadata stores findings in a single `Finding Labels` column.
This script converts that column into 14 binary label columns.

The generated manifest remains outside Git because it contains patient-level
metadata.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

LABEL_COLUMNS = [
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

MANIFEST_COLUMNS = [
    "image_index",
    "image_path",
    "patient_id",
    "follow_up_number",
    "view_position",
    "split",
    *LABEL_COLUMNS,
]


def read_split_names(path: Path) -> set[str]:
    """Read one image filename per line from an official NIH split list."""
    if not path.is_file():
        raise FileNotFoundError(f"Split list not found: {path}")

    names: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            if "," in value or "\t" in value:
                raise ValueError(
                    f"{path.name}:{line_number} must contain one filename per line."
                )
            names.add(value)
    return names


def extract_patient_id(image_index: str) -> str:
    """Extract the NIH patient ID from an image filename.

    Example: 00000013_005.png -> 00000013
    """
    patient_id = image_index.split("_", 1)[0]
    if not patient_id.isdigit():
        raise ValueError(f"Cannot derive numeric patient ID from: {image_index}")
    return patient_id


def parse_finding_labels(raw_value: str) -> dict[str, str]:
    """Convert NIH `Finding Labels` text into binary label columns.

    Examples:
        "No Finding" -> all labels are 0
        "Cardiomegaly" -> Cardiomegaly is 1, all others are 0
        "Infiltration|Mass" -> Infiltration and Mass are 1
    """
    labels = {column: "0" for column in LABEL_COLUMNS}
    raw_value = raw_value.strip()

    if not raw_value or raw_value == "No Finding":
        return labels

    findings = [finding.strip() for finding in raw_value.split("|") if finding.strip()]

    for finding in findings:
        if finding == "No Finding":
            continue
        if finding not in labels:
            raise ValueError(f"Unknown NIH finding label: {finding!r}")
        labels[finding] = "1"

    return labels


def build_manifest(config_path: Path) -> tuple[int, int]:
    """Create the manifest CSV and return train/validation and test row counts."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset = config["dataset"]
    metadata_path = Path(dataset["metadata_file"])
    train_val_path = Path(dataset["train_val_list"])
    test_path = Path(dataset["test_list"])
    images_dir = Path(dataset["images_dir"])
    manifest_path = Path(dataset["manifest_file"])

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    train_val_names = read_split_names(train_val_path)
    test_names = read_split_names(test_path)

    overlap = train_val_names.intersection(test_names)
    if overlap:
        raise ValueError(
            f"Official split lists overlap for {len(overlap)} images; "
            "refusing to build a leaking manifest."
        )

    rows: list[dict[str, str]] = []
    seen_names: set[str] = set()

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "Image Index",
            "Finding Labels",
            "Follow-up #",
            "View Position",
        }
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Metadata is missing required columns: {sorted(missing_columns)}"
            )

        for row in reader:
            image_index = row["Image Index"].strip()
            if image_index in seen_names:
                raise ValueError(f"Duplicate metadata row for image: {image_index}")
            seen_names.add(image_index)

            if image_index in train_val_names:
                split = "train_val"
            elif image_index in test_names:
                split = "test"
            else:
                raise ValueError(
                    f"Image is absent from both official split lists: {image_index}"
                )

            image_path = images_dir / image_index
            if not image_path.is_file():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            rows.append(
                {
                    "image_index": image_index,
                    "image_path": image_path.as_posix(),
                    "patient_id": extract_patient_id(image_index),
                    "follow_up_number": row["Follow-up #"].strip(),
                    "view_position": row["View Position"].strip(),
                    "split": split,
                    **parse_finding_labels(row["Finding Labels"]),
                }
            )

    missing_metadata = (train_val_names | test_names).difference(seen_names)
    if missing_metadata:
        raise ValueError(
            f"{len(missing_metadata)} images listed in split files are missing "
            "from metadata."
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=MANIFEST_COLUMNS, extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)

    train_val_count = sum(row["split"] == "train_val" for row in rows)
    test_count = sum(row["split"] == "test" for row in rows)
    return train_val_count, test_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/classification/config.yaml"),
        help="Path to the classification configuration file.",
    )
    args = parser.parse_args()

    try:
        train_val_count, test_count = build_manifest(args.config)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(f"PASS: manifest written for configuration {args.config}")
    print(f"Train/validation images: {train_val_count}")
    print(f"Held-out test images: {test_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())