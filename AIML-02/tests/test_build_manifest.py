"""Tests for NIH ChestX-ray14 manifest construction."""

from pathlib import Path

import pytest
import yaml

from ml.classification.build_manifest import (
    extract_patient_id,
    parse_finding_labels,
    read_split_names,
)


def test_extract_patient_id_from_nih_filename():
    assert extract_patient_id("00000013_005.png") == "00000013"


def test_extract_patient_id_rejects_invalid_filename():
    with pytest.raises(ValueError):
        extract_patient_id("invalid-image.png")


def test_parse_no_finding_sets_all_labels_to_zero():
    labels = parse_finding_labels("No Finding")
    assert set(labels.values()) == {"0"}
    assert len(labels) == 14


def test_parse_single_finding():
    labels = parse_finding_labels("Cardiomegaly")
    assert labels["Cardiomegaly"] == "1"
    assert labels["Atelectasis"] == "0"
    assert sum(value == "1" for value in labels.values()) == 1


def test_parse_multiple_findings():
    labels = parse_finding_labels("Infiltration|Mass")
    assert labels["Infiltration"] == "1"
    assert labels["Mass"] == "1"
    assert labels["Cardiomegaly"] == "0"
    assert sum(value == "1" for value in labels.values()) == 2


def test_parse_unknown_finding_rejected():
    with pytest.raises(ValueError, match="Unknown NIH finding label"):
        parse_finding_labels("NotARealFinding")


def test_read_split_names_rejects_csv_like_file(tmp_path: Path):
    split_file = tmp_path / "bad_list.txt"
    split_file.write_text("image1.png,image2.png\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_split_names(split_file)


def test_manifest_builder_rejects_overlapping_splits(tmp_path: Path):
    from ml.classification import build_manifest as manifest_module

    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "Image Index,Finding Labels,Follow-up #,View Position\n"
        "00000001_000.png,No Finding,0,PA\n",
        encoding="utf-8",
    )

    train_val = tmp_path / "train_val_list.txt"
    test = tmp_path / "test_list.txt"
    train_val.write_text("00000001_000.png\n", encoding="utf-8")
    test.write_text("00000001_000.png\n", encoding="utf-8")

    config = {
        "dataset": {
            "metadata_file": str(metadata),
            "train_val_list": str(train_val),
            "test_list": str(test),
            "images_dir": str(tmp_path),
            "manifest_file": str(tmp_path / "manifest.csv"),
        }
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        manifest_module.build_manifest(config_path)