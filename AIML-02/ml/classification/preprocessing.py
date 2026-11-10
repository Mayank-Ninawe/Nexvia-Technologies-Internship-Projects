"""Reusable preprocessing for NIH ChestX-ray14 classification images.

This module intentionally keeps preprocessing independent from model code so
that the same logic can be used for training, evaluation, and later API
inference.

Educational use only; not for clinical use.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_MEAN = 0.5
DEFAULT_STD = 0.5


class ImagePreprocessingError(ValueError):
    """Raised when an image cannot be safely preprocessed."""


def load_grayscale_image(image_path: Path) -> Image.Image:
    """Load an image and convert it to single-channel grayscale."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if image_path.stat().st_size == 0:
        raise ImagePreprocessingError(f"Image file is empty: {image_path}")

    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source)
            return image.convert("L")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImagePreprocessingError(
            f"Cannot read image {image_path}: {type(error).__name__}"
        ) from error


def resize_image(
    image: Image.Image,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Image.Image:
    """Resize an image to the configured width and height."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Image size must contain positive dimensions.")

    return image.resize((width, height), resample=Image.Resampling.BILINEAR)


def normalize_image(
    image: Image.Image,
    mean: float = DEFAULT_MEAN,
    std: float = DEFAULT_STD,
) -> np.ndarray:
    """Convert a grayscale image to a normalized float32 NumPy array.

    Output shape: (1, height, width).
    Output values are approximately centered at zero when mean and std are 0.5.
    """
    if std <= 0:
        raise ValueError("Normalization standard deviation must be positive.")

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - mean) / std
    return array[np.newaxis, :, :]


def augment_image(image: Image.Image, random_seed: int | None = None) -> Image.Image:
    """Apply bounded augmentation suitable for chest X-rays.

    Currently uses a horizontal flip with probability 0.5. This preserves the
    anatomical plausibility of a frontal chest radiograph while increasing
    apparent sample variation.
    """
    rng = random.Random(random_seed)
    if rng.random() < 0.5:
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image.copy()


def preprocess_image(
    image_path: Path,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: float = DEFAULT_MEAN,
    std: float = DEFAULT_STD,
    augment: bool = False,
    random_seed: int | None = None,
) -> np.ndarray:
    """Load, optionally augment, resize, and normalize one image."""
    image = load_grayscale_image(image_path)

    if augment:
        image = augment_image(image, random_seed=random_seed)

    image = resize_image(image, size=size)
    return normalize_image(image, mean=mean, std=std)