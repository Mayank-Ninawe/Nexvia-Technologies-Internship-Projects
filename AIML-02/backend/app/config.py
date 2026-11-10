"""Backend configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def max_upload_size_bytes() -> int:
    """Return the configured upload limit in bytes."""
    raw_value = os.getenv("MAX_UPLOAD_SIZE_MB", "50")
    try:
        megabytes = float(raw_value)
    except ValueError as error:
        raise ValueError("MAX_UPLOAD_SIZE_MB must be a positive number.") from error
    if megabytes <= 0:
        raise ValueError("MAX_UPLOAD_SIZE_MB must be a positive number.")
    return int(megabytes * 1024 * 1024)


def classification_checkpoint_path() -> Path:
    """Return the configured classification checkpoint path."""
    configured = os.getenv(
        "CLASSIFICATION_MODEL_PATH",
        "models/classification/best_classifier.pt",
    )
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path
