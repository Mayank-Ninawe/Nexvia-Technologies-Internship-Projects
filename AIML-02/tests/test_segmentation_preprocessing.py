"""Tests for MSD Task09 Spleen preprocessing and alignment."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from ml.segmentation.preprocessing import (
    IMAGE_KEY,
    LABEL_KEY,
    SegmentationDataError,
    create_dataset,
    list_volume_pairs,
    validate_sample,
)


@pytest.fixture
def spleen_config():
    return {
        "dataset": {
            "images_dir": "",
            "labels_dir": "",
        },
        "preprocessing": {
            "target_orientation": "RAS",
            "target_spacing": [1.5, 1.5, 2.0],
            "intensity_a_min": -175.0,
            "intensity_a_max": 250.0,
            "intensity_b_min": 0.0,
            "intensity_b_max": 1.0,
            "roi_crop_source": "image",
        },
        "training": {
            "num_workers": 0,
        },
    }


def create_nifti_pair(directory: Path, stem: str, size=(16, 16, 12)) -> None:
    image_data = np.random.rand(*size).astype(np.float32) * 200.0 - 100.0
    label_data = np.zeros(size, dtype=np.uint8)
    label_data[4:8, 4:8, 4:8] = 1

    affine = np.eye(4)
    image = nib.Nifti1Image(image_data, affine)
    label = nib.Nifti1Image(label_data, affine)

    nib.save(image, directory / "images" / f"{stem}.nii.gz")
    nib.save(label, directory / "labels" / f"{stem}.nii.gz")


def test_list_volume_pairs_matches_filenames(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    create_nifti_pair(tmp_path, "spleen_01")
    create_nifti_pair(tmp_path, "spleen_02")

    pairs = list_volume_pairs(images_dir, labels_dir)

    assert len(pairs) == 2
    assert Path(pairs[0][IMAGE_KEY]).name == "spleen_01.nii.gz"
    assert Path(pairs[0][LABEL_KEY]).name == "spleen_01.nii.gz"


def test_list_volume_pairs_rejects_mismatched_counts(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    create_nifti_pair(tmp_path, "spleen_01")
    (labels_dir / "extra.nii.gz").write_bytes(b"")

    with pytest.raises(SegmentationDataError, match="images but"):
        list_volume_pairs(images_dir, labels_dir)


def test_preprocessing_preserves_image_mask_alignment(tmp_path: Path, spleen_config):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    create_nifti_pair(tmp_path, "spleen_01")
    spleen_config["dataset"]["images_dir"] = str(images_dir)
    spleen_config["dataset"]["labels_dir"] = str(labels_dir)

    dataset = create_dataset(spleen_config, limit=1)
    sample = dataset[0]

    validate_sample(sample)

    assert sample[IMAGE_KEY].shape[0] == 1
    assert sample[LABEL_KEY].shape[0] == 1
    assert sample[IMAGE_KEY].shape[1:] == sample[LABEL_KEY].shape[1:]
    assert torch.isfinite(sample[IMAGE_KEY]).all()
    assert set(torch.unique(sample[LABEL_KEY]).tolist()).issubset({0.0, 1.0})


def test_create_dataset_supports_non_overlapping_offset(tmp_path: Path, spleen_config):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    create_nifti_pair(tmp_path, "spleen_01")
    create_nifti_pair(tmp_path, "spleen_02")
    spleen_config["dataset"]["images_dir"] = str(images_dir)
    spleen_config["dataset"]["labels_dir"] = str(labels_dir)

    dataset = create_dataset(spleen_config, limit=1, offset=1)

    assert Path(dataset.data[0][IMAGE_KEY]).name == "spleen_02.nii.gz"


def test_validate_sample_rejects_misaligned_shapes():
    sample = {
        IMAGE_KEY: torch.randn(1, 16, 16, 12),
        LABEL_KEY: torch.zeros(1, 16, 16, 10),
    }

    with pytest.raises(SegmentationDataError, match="does not match"):
        validate_sample(sample)


def test_validate_sample_rejects_non_binary_label():
    sample = {
        IMAGE_KEY: torch.randn(1, 16, 16, 12),
        LABEL_KEY: torch.full((1, 16, 16, 12), 2.0),
    }

    with pytest.raises(SegmentationDataError, match="non-binary"):
        validate_sample(sample)