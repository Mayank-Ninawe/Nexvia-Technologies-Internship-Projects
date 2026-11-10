"""Tests for NIH classification image preprocessing."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml.classification.preprocessing import (
    ImagePreprocessingError,
    augment_image,
    load_grayscale_image,
    normalize_image,
    preprocess_image,
    resize_image,
)


def create_image(path: Path, size: tuple[int, int] = (64, 48)) -> Path:
    Image.new("L", size, color=128).save(path)
    return path


def test_load_grayscale_image_converts_rgb_to_l(tmp_path: Path):
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_path)

    image = load_grayscale_image(image_path)

    assert image.mode == "L"
    assert image.size == (32, 32)


def test_load_grayscale_image_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_grayscale_image(tmp_path / "missing.png")


def test_load_grayscale_image_rejects_empty_file(tmp_path: Path):
    image_path = tmp_path / "empty.png"
    image_path.write_bytes(b"")

    with pytest.raises(ImagePreprocessingError, match="empty"):
        load_grayscale_image(image_path)


def test_load_grayscale_image_rejects_corrupt_file(tmp_path: Path):
    image_path = tmp_path / "corrupt.png"
    image_path.write_bytes(b"not-an-image")

    with pytest.raises(ImagePreprocessingError, match="Cannot read image"):
        load_grayscale_image(image_path)


def test_resize_image_returns_configured_shape():
    image = Image.new("L", (100, 50))
    resized = resize_image(image, size=(224, 224))

    assert resized.size == (224, 224)


def test_resize_image_rejects_invalid_size():
    image = Image.new("L", (32, 32))

    with pytest.raises(ValueError):
        resize_image(image, size=(0, 224))


def test_normalize_image_shape_and_range():
    image = Image.new("L", (32, 32), color=128)
    normalized = normalize_image(image)

    assert normalized.shape == (1, 32, 32)
    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert normalized.min() >= -1.0
    assert normalized.max() <= 1.0


def test_normalize_image_rejects_non_positive_std():
    image = Image.new("L", (8, 8))

    with pytest.raises(ValueError):
        normalize_image(image, std=0.0)


def test_augment_image_is_reproducible_with_seed():
    image = Image.new("L", (16, 16), color=128)

    first = augment_image(image, random_seed=42)
    second = augment_image(image, random_seed=42)

    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_preprocess_image_returns_expected_shape(tmp_path: Path):
    image_path = create_image(tmp_path / "image.png")

    result = preprocess_image(
        image_path,
        size=(224, 224),
        augment=False,
        random_seed=42,
    )

    assert result.shape == (1, 224, 224)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_preprocess_image_rejects_invalid_input(tmp_path: Path):
    image_path = tmp_path / "invalid.png"
    image_path.write_bytes(b"invalid")

    with pytest.raises(ImagePreprocessingError):
        preprocess_image(image_path)