"""Evaluate a trusted classifier checkpoint on validation data only.

Educational use only; not for clinical use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ml.classification.dataset import LABEL_COLUMNS, ChestXrayDataset
from ml.classification.evaluate import collect_predictions, compute_metrics, load_model
from ml.run_metadata import build_run_metadata


def sha256_file(path: Path) -> str:
    """Compute a file hash without loading the complete file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> dict:
    """Measure a checkpoint on the full configured validation split."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path}")

    started = time.perf_counter()

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    manifest_path = Path(config["dataset"]["manifest_file"]).with_name(
        "nih_chestxray14_patient_splits.csv"
    )
    image_config = config["image"]
    training_config = config["training"]

    checkpoint_hash = sha256_file(checkpoint_path)
    manifest_hash = sha256_file(manifest_path)
    config_hash = sha256_file(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(checkpoint_path, device)

    stored_labels = checkpoint.get("label_columns")
    if stored_labels is not None and list(stored_labels) != list(LABEL_COLUMNS):
        raise ValueError("Checkpoint label order does not match LABEL_COLUMNS.")

    dataset = ChestXrayDataset(
        manifest_path=manifest_path,
        split="validation",
        image_size=(image_config["resize_width"], image_config["resize_height"]),
        mean=image_config["normalize_mean"],
        std=image_config["normalize_std"],
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    print(
        f"Validating {checkpoint_path} on {len(dataset)} images using {device}.",
        flush=True,
    )
    logits, targets = collect_predictions(model, loader, device)
    metrics = compute_metrics(logits, targets, threshold=0.5)

    sample_count = metrics.pop("test_samples")
    metrics["validation_samples"] = sample_count
    metrics["split"] = "validation"
    metrics["device"] = str(device)
    metrics["checkpoint_path"] = checkpoint_path.as_posix()
    metrics["checkpoint_sha256"] = checkpoint_hash
    metrics["manifest_sha256"] = manifest_hash
    metrics["config_sha256"] = config_hash
    metrics["label_columns"] = list(LABEL_COLUMNS)
    metrics["config_snapshot"] = config
    metrics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    metrics["run_metadata"] = build_run_metadata(
        command=sys.argv,
        config_path=config_path,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
    )

    if sha256_file(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during validation.")
    if sha256_file(manifest_path) != manifest_hash:
        raise RuntimeError("Manifest changed during validation.")
    if sha256_file(config_path) != config_hash:
        raise RuntimeError("Configuration changed during validation.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/classification/config.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        result = validate_checkpoint(args.config, args.checkpoint, args.output)
    except (OSError, ValueError, KeyError, RuntimeError, yaml.YAMLError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(f"Validation samples: {result['validation_samples']}")
    print(f"Macro AUROC: {result['macro_auroc']}")
    print(f"Macro average precision: {result['macro_average_precision']}")
    print(f"Elapsed seconds: {result['elapsed_seconds']}")
    print(f"Checkpoint SHA256: {result['checkpoint_sha256']}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())