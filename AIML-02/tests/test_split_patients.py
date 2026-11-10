"""Tests for leakage-safe NIH patient-level splitting."""

import csv
from pathlib import Path

import pytest

from ml.classification.split_patients import (
    assign_patient_splits,
    create_patient_split,
)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "image_index",
        "image_path",
        "patient_id",
        "follow_up_number",
        "view_position",
        "split",
        "Atelectasis",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_row(
    image_index: str,
    patient_id: str,
    split: str,
) -> dict[str, str]:
    return {
        "image_index": image_index,
        "image_path": f"images/{image_index}",
        "patient_id": patient_id,
        "follow_up_number": "0",
        "view_position": "PA",
        "split": split,
        "Atelectasis": "0",
    }


def test_assign_patient_splits_is_reproducible():
    rows = [
        make_row(f"{index:08d}_000.png", f"{index:08d}", "train_val")
        for index in range(20)
    ]

    first = assign_patient_splits(rows, 0.25, 42)
    second = assign_patient_splits(rows, 0.25, 42)

    assert first == second


def test_assign_patient_splits_creates_validation_patients():
    rows = [
        make_row(f"{index:08d}_000.png", f"{index:08d}", "train_val")
        for index in range(100)
    ]

    assignments = assign_patient_splits(rows, 0.2, 42)

    assert len(assignments) == 100
    assert sum(split == "validation" for split in assignments.values()) == 20
    assert sum(split == "train" for split in assignments.values()) == 80


def test_assign_patient_splits_rejects_invalid_fraction():
    rows = [make_row("00000001_000.png", "00000001", "train_val")]

    with pytest.raises(ValueError):
        assign_patient_splits(rows, 0.0, 42)

    with pytest.raises(ValueError):
        assign_patient_splits(rows, 1.0, 42)


def test_create_patient_split_prevents_patient_leakage(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    rows = [
        make_row("00000001_000.png", "00000001", "train_val"),
        make_row("00000001_001.png", "00000001", "train_val"),
        make_row("00000002_000.png", "00000002", "train_val"),
        make_row("00000003_000.png", "00000003", "test"),
    ]
    write_manifest(manifest, rows)

    output = tmp_path / "splits.csv"
    counts = create_patient_split(
        manifest_path=manifest,
        output_path=output,
        validation_fraction=0.5,
        random_seed=42,
    )

    with output.open("r", encoding="utf-8", newline="") as handle:
        result_rows = list(csv.DictReader(handle))

    patient_splits: dict[str, set[str]] = {}
    for row in result_rows:
        patient_splits.setdefault(row["patient_id"], set()).add(row["split"])

    assert all(len(splits) == 1 for splits in patient_splits.values())
    assert counts["test"] == 1
    assert counts["train"] + counts["validation"] == 3


def test_create_patient_split_preserves_official_test(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    rows = [
        make_row("00000001_000.png", "00000001", "train_val"),
        make_row("00000002_000.png", "00000002", "test"),
    ]
    write_manifest(manifest, rows)

    output = tmp_path / "splits.csv"
    create_patient_split(
        manifest_path=manifest,
        output_path=output,
        validation_fraction=0.5,
        random_seed=42,
    )

    with output.open("r", encoding="utf-8", newline="") as handle:
        result_rows = list(csv.DictReader(handle))

    test_rows = [row for row in result_rows if row["split"] == "test"]
    assert len(test_rows) == 1
    assert test_rows[0]["image_index"] == "00000002_000.png"