"""MONAI preprocessing for MSD Task09 Spleen CT volumes and masks.

The pipeline preserves the spatial relationship between each CT volume and its
spleen segmentation mask.

Educational use only; not for clinical use.
"""

from __future__ import annotations

from pathlib import Path

import torch
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

IMAGE_KEY = "image"
LABEL_KEY = "label"


class SegmentationDataError(ValueError):
    """Raised when a CT volume and mask are misaligned or invalid."""


def build_preprocessing_transforms(config: dict) -> Compose:
    """Build deterministic MONAI transforms for image-mask pairs."""
    preprocessing = config["preprocessing"]

    return Compose(
        [
            LoadImaged(keys=[IMAGE_KEY, LABEL_KEY]),
            EnsureChannelFirstd(keys=[IMAGE_KEY, LABEL_KEY]),
            Orientationd(
                keys=[IMAGE_KEY, LABEL_KEY],
                axcodes=preprocessing["target_orientation"],
            ),
            Spacingd(
                keys=[IMAGE_KEY, LABEL_KEY],
                pixdim=preprocessing["target_spacing"],
                mode=("bilinear", "nearest"),
                align_corners=True,
            ),
            ScaleIntensityRanged(
                keys=[IMAGE_KEY],
                a_min=preprocessing["intensity_a_min"],
                a_max=preprocessing["intensity_a_max"],
                b_min=preprocessing["intensity_b_min"],
                b_max=preprocessing["intensity_b_max"],
                clip=True,
            ),
            CropForegroundd(
                keys=[IMAGE_KEY, LABEL_KEY],
                source_key=IMAGE_KEY,
                allow_smaller=True,
            ),
            ToTensord(keys=[IMAGE_KEY, LABEL_KEY]),
        ]
    )


def validate_sample(sample: dict) -> None:
    """Validate that a preprocessed CT/mask pair remains aligned."""
    image = sample[IMAGE_KEY]
    label = sample[LABEL_KEY]

    if not isinstance(image, torch.Tensor) or not isinstance(label, torch.Tensor):
        raise SegmentationDataError("Image and label must both be tensors.")

    if image.shape[1:] != label.shape[1:]:
        raise SegmentationDataError(
            f"Image spatial shape {tuple(image.shape[1:])} does not match "
            f"label spatial shape {tuple(label.shape[1:])}."
        )

    if not torch.isfinite(image).all():
        raise SegmentationDataError("CT image contains non-finite values.")

    unique_labels = torch.unique(label)
    invalid_values = unique_labels[~torch.isin(unique_labels, torch.tensor([0.0, 1.0]))]
    if invalid_values.numel():
        raise SegmentationDataError(
            f"Label contains non-binary values: {invalid_values.tolist()}"
        )


def list_volume_pairs(images_dir: Path, labels_dir: Path) -> list[dict[str, str]]:
    """Create aligned image/label file pairs, ignoring macOS metadata files."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    image_files = sorted(
        path
        for path in images_dir.glob("*.nii.gz")
        if not path.name.startswith("._")
    )
    label_files = sorted(
        path
        for path in labels_dir.glob("*.nii.gz")
        if not path.name.startswith("._")
    )

    if len(image_files) != len(label_files):
        raise SegmentationDataError(
            f"Found {len(image_files)} images but {len(label_files)} labels."
        )

    samples = []
    for image_path, label_path in zip(image_files, label_files, strict=True):
        image_stem = image_path.name.replace(".nii.gz", "")
        label_stem = label_path.name.replace(".nii.gz", "")

        if image_stem != label_stem:
            raise SegmentationDataError(
                f"Image/mask filename mismatch: {image_path.name} and {label_path.name}."
            )

        samples.append(
            {
                IMAGE_KEY: str(image_path),
                LABEL_KEY: str(label_path),
            }
        )

    return samples


def create_dataset(
    config: dict,
    limit: int | None = None,
    offset: int = 0,
) -> CacheDataset:
    """Create a cached MONAI dataset for bounded spleen segmentation pilots."""
    dataset_config = config["dataset"]
    samples = list_volume_pairs(
        images_dir=Path(dataset_config["images_dir"]),
        labels_dir=Path(dataset_config["labels_dir"]),
    )

    if offset < 0:
        raise ValueError("offset must not be negative.")
    if offset:
        samples = samples[offset:]

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")
        samples = samples[:limit]

    if not samples:
        raise SegmentationDataError("No valid image/label pairs were found.")

    return CacheDataset(
        data=samples,
        transform=build_preprocessing_transforms(config),
        cache_rate=1.0,
        num_workers=config["training"]["num_workers"],
    )