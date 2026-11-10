"""Tests for bounded NIH image validation."""

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from ml.classification.validate_images import (
    read_manifest,
    validate_image,
    validate_manifest_images,
)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["image_index", "image_path", "split"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_read_manifest_respects_limit(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    rows = [
        {
            "image_index": f"{index:08d}_000.png",
            "image_path": str(tmp_path / f"{index}.png"),
            "split": "train_val",
        }
        for index in range(10)
    ]
    write_manifest(manifest, rows)

    selected = read_manifest(manifest, limit=3)

    assert len(selected) == 3
    assert selected[0]["image_index"] == "00000000_000.png"


def test_validate_image_accepts_valid_png(tmp_path: Path):
    image_path = tmp_path / "valid.png"
    Image.new("L", (32, 32), color=128).save(image_path)

    result = validate_image(image_path)

    assert result["valid"] is True
    assert result["width"] == 32
    assert result["height"] == 32
    assert result["mode"] == "L"
    assert result["error"] is None


def test_validate_image_rejects_missing_file(tmp_path: Path):
    result = validate_image(tmp_path / "missing.png")

    assert result["valid"] is False
    assert result["error"] == "file_not_found"


def test_validate_image_rejects_empty_file(tmp_path: Path):
    image_path = tmp_path / "empty.png"
    image_path.write_bytes(b"")

    result = validate_image(image_path)

    assert result["valid"] is False
    assert result["error"] == "empty_file"


def test_validate_image_rejects_corrupt_file(tmp_path: Path):
    image_path = tmp_path / "corrupt.png"
    image_path.write_bytes(b"this is not a png image")

    result = validate_image(image_path)

    assert result["valid"] is False
    assert result["error"] is not None
    assert result["error"].startswith("unreadable_image")


def test_validate_manifest_images_writes_report(tmp_path: Path):
    valid_image = tmp_path / "valid.png"
    Image.new("L", (16, 16), color=0).save(valid_image)

    corrupt_image = tmp_path / "corrupt.png"
    corrupt_image.write_bytes(b"not-an-image")

    manifest = tmp_path / "manifest.csv"
    rows = [
        {
            "image_index": "valid.png",
            "image_path": str(valid_image),
            "split": "train_val",
        },
        {
            "image_index": "corrupt.png",
            "image_path": str(corrupt_image),
            "split": "test",
        },
    ]
    write_manifest(manifest, rows)

    output = tmp_path / "report.json"
    report = validate_manifest_images(manifest, output, limit=None)

    assert report["validated_count"] == 2
    assert report["valid_count"] == 1
    assert report["invalid_count"] == 1
    assert output.is_file()

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["valid_count"] == 1
    assert saved["invalid_count"] == 1


def test_validate_manifest_rejects_missing_required_column(tmp_path: Path):
    manifest = tmp_path / "bad_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_index"])
        writer.writeheader()
        writer.writerow({"image_index": "00000001_000.png"})

    with pytest.raises(ValueError, match="missing required columns"):
        read_manifest(manifest)