"""Reproducibility metadata helpers for ML training and evaluation runs."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


def build_run_metadata(
    *,
    command: list[str],
    config_path: Path,
    manifest_path: Path | None = None,
    checkpoint_path: Path | None = None,
    checkpoint: dict | None = None,
) -> dict:
    """Return measured environment and artifact provenance for a run."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    metadata = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "git_revision": revision,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "config_path": config_path.as_posix(),
    }

    try:
        metadata["monai_version"] = version("monai")
    except PackageNotFoundError as error:
        raise RuntimeError("Unable to determine the installed MONAI version.") from error

    if manifest_path is not None:
        metadata["manifest_path"] = manifest_path.as_posix()
    if checkpoint_path is not None:
        metadata["checkpoint_path"] = checkpoint_path.as_posix()
    if checkpoint is not None:
        for field in ("epoch", "validation_loss", "best_validation_loss", "validation_dice"):
            if field in checkpoint:
                metadata[field] = checkpoint[field]

    return metadata


def write_run_metadata(output_path: Path, metadata: dict) -> None:
    """Write metadata as a standalone JSON document beside a run output."""
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
