"""Evaluate the trained NIH ChestX-ray14 classifier on the held-out test set.

Educational use only; not for clinical use.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from ml.classification.dataset import LABEL_COLUMNS, ChestXrayDataset
from ml.classification.pilot import create_model
from ml.run_metadata import build_run_metadata


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Load the best trained classifier checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = create_model(num_classes=len(LABEL_COLUMNS), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect logits and targets for the full evaluation set."""
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(images)

            all_logits.append(logits.float().cpu())
            all_targets.append(targets.cpu())

    return torch.cat(all_logits), torch.cat(all_targets)


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """Compute per-label and macro metrics for a multi-label classifier."""
    probabilities = torch.sigmoid(logits).numpy()
    binary_targets = targets.numpy()
    binary_predictions = (probabilities >= threshold).astype(int)

    per_label = {}

    for index, label in enumerate(LABEL_COLUMNS):
        label_targets = binary_targets[:, index]
        label_probabilities = probabilities[:, index]
        label_predictions = binary_predictions[:, index]
        positive_count = int(label_targets.sum())
        has_both_classes = len(set(label_targets.tolist())) == 2

        if positive_count == 0:
            per_label[label] = {
                "support_positive": 0,
                "prevalence": 0.0,
                "auroc": None,
                "average_precision": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "note": "No positive test samples; ranking metrics undefined.",
            }
            continue

        label_metrics = {
            "support_positive": positive_count,
            "prevalence": round(float(label_targets.mean()), 6),
            "average_precision": round(
                float(average_precision_score(label_targets, label_probabilities)), 6
            ),
            "precision": round(
                float(precision_score(label_targets, label_predictions, zero_division=0)),
                6,
            ),
            "recall": round(
                float(recall_score(label_targets, label_predictions, zero_division=0)),
                6,
            ),
            "f1": round(
                float(f1_score(label_targets, label_predictions, zero_division=0)),
                6,
            ),
        }
        if has_both_classes:
            label_metrics["auroc"] = round(
                float(roc_auc_score(label_targets, label_probabilities)), 6
            )
        else:
            label_metrics["auroc"] = None
            label_metrics["note"] = (
                "Only one target class is present; AUROC is undefined for this subset."
            )

        per_label[label] = label_metrics

    defined_auroc = [
        metrics["auroc"]
        for metrics in per_label.values()
        if metrics["auroc"] is not None
    ]
    defined_ap = [
        metrics["average_precision"]
        for metrics in per_label.values()
        if metrics["average_precision"] is not None
    ]

    return {
        "threshold": threshold,
        "test_samples": int(targets.shape[0]),
        "macro_auroc": round(sum(defined_auroc) / len(defined_auroc), 6)
        if defined_auroc
        else None,
        "macro_average_precision": round(sum(defined_ap) / len(defined_ap), 6)
        if defined_ap
        else None,
        "per_label": per_label,
    }


def evaluate_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    threshold: float,
    limit: int | None,
    command: list[str] | None = None,
) -> dict:
    """Evaluate a checkpoint using the official held-out test split."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset_config = config["dataset"]
    image_config = config["image"]
    training_config = config["training"]

    split_manifest = Path(dataset_config["manifest_file"]).with_name(
        "nih_chestxray14_patient_splits.csv"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(checkpoint_path, device)

    test_dataset = ChestXrayDataset(
        manifest_path=split_manifest,
        split="test",
        image_size=(image_config["resize_width"], image_config["resize_height"]),
        mean=image_config["normalize_mean"],
        std=image_config["normalize_std"],
        augment=False,
        limit=limit,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    logits, targets = collect_predictions(model, loader, device)
    metrics = compute_metrics(logits, targets, threshold=threshold)
    metrics["checkpoint_path"] = checkpoint_path.as_posix()
    metrics["device"] = str(device)
    metrics["run_metadata"] = build_run_metadata(
        command=command or ["programmatic evaluation"],
        config_path=config_path,
        manifest_path=split_manifest,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "support_positive",
                "prevalence",
                "auroc",
                "average_precision",
                "precision",
                "recall",
                "f1",
                "note",
            ],
        )
        writer.writeheader()
        for label, label_metrics in metrics["per_label"].items():
            writer.writerow({"label": label, **label_metrics})

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/classification/config.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/classification/best_classifier.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/classification_test_metrics.json"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        print("FAIL: threshold must be between 0 and 1.", file=sys.stderr)
        return 1

    try:
        metrics = evaluate_checkpoint(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            threshold=args.threshold,
            limit=args.limit,
            command=sys.argv,
        )
    except (OSError, ValueError, KeyError, RuntimeError, yaml.YAMLError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print("PASS: classification evaluation completed.")
    print(f"Test samples: {metrics['test_samples']}")
    print(f"Macro AUROC: {metrics['macro_auroc']}")
    print(f"Macro average precision: {metrics['macro_average_precision']}")
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())