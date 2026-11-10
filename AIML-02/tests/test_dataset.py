"""Tests for the NIH ChestX-ray14 PyTorch dataset."""

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image

from ml.classification.dataset import LABEL_COLUMNS, ChestXrayDataset


def write_split_manifest(path: Path) -> Path:
    fieldnames = [
        "image_index",
        "image_path",
        "patient_id",
        "follow_up_number",
        "view_position",
        "split",
        *LABEL_COLUMNS,
    ]

    rows = []
    for index, split in enumerate(["train", "validation", "test"]):
        image_path = path / f"{index}.png"
        Image.new("L", (32, 32), color=128).save(image_path)
        rows.append(
            {
                "image_index": f"{index}.png",
                "image_path": str(image_path),
                "patient_id": f"{index:08d}",
                "follow_up_number": "0",
                "view_position": "PA",
                "split": split,
                **{column: "0" for column in LABEL_COLUMNS},
            }
        )
        rows[-1]["Cardiomegaly"] = "1"

    manifest = path / "splits.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return manifest


def test_dataset_filters_requested_split(tmp_path: Path):
    manifest = write_split_manifest(tmp_path)

    dataset = ChestXrayDataset(manifest, split="train")

    assert len(dataset) == 1


def test_dataset_returns_expected_tensor_shapes(tmp_path: Path):
    manifest = write_split_manifest(tmp_path)

    dataset = ChestXrayDataset(manifest, split="train", image_size=(64, 64))
    image, target = dataset[0]

    assert image.shape == (1, 64, 64)
    assert image.dtype == torch.float32
    assert target.shape == (len(LABEL_COLUMNS),)
    assert target.dtype == torch.float32
    assert target[1].item() == 1.0
    assert torch.isfinite(image).all()


def test_dataset_rejects_invalid_split(tmp_path: Path):
    manifest = write_split_manifest(tmp_path)

    with pytest.raises(ValueError, match="split must be"):
        ChestXrayDataset(manifest, split="unknown")


def test_dataset_rejects_missing_manifest(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ChestXrayDataset(tmp_path / "missing.csv", split="train")


def test_create_model_outputs_expected_shape():
    from ml.classification.pilot import create_model

    model = create_model(num_classes=len(LABEL_COLUMNS), pretrained=False)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 1, 224, 224))

    assert output.shape == (2, len(LABEL_COLUMNS))
    assert torch.isfinite(output).all()