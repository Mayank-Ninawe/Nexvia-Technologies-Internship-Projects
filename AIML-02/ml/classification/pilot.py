"""Run a bounded NIH ChestX-ray14 classification pilot.

This pilot verifies that the data pipeline, model, GPU execution, loss
calculation, and backward pass work before any longer training run.

Educational use only; not for clinical use.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from ml.classification.dataset import LABEL_COLUMNS, ChestXrayDataset


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a DenseNet-121 classifier adapted to grayscale input."""
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)

    original_conv = model.features.conv0
    model.features.conv0 = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False,
    )

    if pretrained:
        with torch.no_grad():
            rgb_weights = original_conv.weight
            grayscale_weights = rgb_weights.mean(dim=1, keepdim=True)
            model.features.conv0.weight.copy_(grayscale_weights)

    model.classifier = nn.Linear(
        in_features=model.classifier.in_features,
        out_features=num_classes,
    )
    return model


def build_data_loaders(
    config: dict,
    train_limit: int,
    validation_limit: int,
) -> tuple[DataLoader, DataLoader]:
    """Create bounded train and validation DataLoaders."""
    dataset_config = config["dataset"]
    image_config = config["image"]
    training_config = config["training"]

    split_manifest = Path(dataset_config["manifest_file"]).with_name(
        "nih_chestxray14_patient_splits.csv"
    )

    train_dataset = ChestXrayDataset(
        manifest_path=split_manifest,
        split="train",
        image_size=(image_config["resize_width"], image_config["resize_height"]),
        mean=image_config["normalize_mean"],
        std=image_config["normalize_std"],
        augment=True,
        limit=train_limit,
    )

    validation_dataset = ChestXrayDataset(
        manifest_path=split_manifest,
        split="validation",
        image_size=(image_config["resize_width"], image_config["resize_height"]),
        mean=image_config["normalize_mean"],
        std=image_config["normalize_std"],
        augment=False,
        limit=validation_limit,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, validation_loader


def run_pilot(
    config_path: Path,
    output_path: Path,
    train_limit: int,
    validation_limit: int,
    epochs: int = 1,
) -> dict:
    """Run one bounded training/validation pass and return measured results."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, validation_loader = build_data_loaders(
        config,
        train_limit=train_limit,
        validation_limit=validation_limit,
    )

    model = create_model(num_classes=len(LABEL_COLUMNS), pretrained=True)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    train_loss_total = 0.0
    train_batches = 0
    start_time = time.perf_counter()

    model.train()
    for epoch in range(epochs):
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += float(loss.detach().cpu())
            train_batches += 1

    elapsed_seconds = time.perf_counter() - start_time

    validation_loss_total = 0.0
    validation_batches = 0

    model.eval()
    with torch.no_grad():
        for images, targets in validation_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            validation_loss_total += float(loss.detach().cpu())
            validation_batches += 1

    report = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "model": "densenet121",
        "num_classes": len(LABEL_COLUMNS),
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "batch_size": config["training"]["batch_size"],
        "epochs": epochs,
        "train_batches": train_batches,
        "validation_batches": validation_batches,
        "mean_train_loss": train_loss_total / max(train_batches, 1),
        "mean_validation_loss": validation_loss_total / max(validation_batches, 1),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "samples_per_second": round(
            (len(train_loader.dataset) * epochs) / max(elapsed_seconds, 1e-9),
            3,
        ),
        "gpu_memory_allocated_mb": round(
            torch.cuda.memory_allocated() / 1024**2, 2
        )
        if torch.cuda.is_available()
        else None,
        "gpu_memory_reserved_mb": round(
            torch.cuda.memory_reserved() / 1024**2, 2
        )
        if torch.cuda.is_available()
        else None,
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        report["gpu_peak_memory_allocated_mb"] = round(
            torch.cuda.max_memory_allocated() / 1024**2, 2
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/classification/config.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/classification_pilot.json"),
    )
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--validation-limit", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    if args.train_limit <= 0 or args.validation_limit <= 0 or args.epochs <= 0:
        print("FAIL: limits and epochs must be greater than zero.", file=sys.stderr)
        return 1

    try:
        report = run_pilot(
            config_path=args.config,
            output_path=args.output,
            train_limit=args.train_limit,
            validation_limit=args.validation_limit,
            epochs=args.epochs,
        )
    except (OSError, ValueError, KeyError, RuntimeError, yaml.YAMLError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print("PASS: classification pilot completed.")
    print(f"Device: {report['device']}")
    print(f"Train samples: {report['train_samples']}")
    print(f"Validation samples: {report['validation_samples']}")
    print(f"Mean train loss: {report['mean_train_loss']:.4f}")
    print(f"Mean validation loss: {report['mean_validation_loss']:.4f}")
    print(f"Elapsed seconds: {report['elapsed_seconds']}")
    print(f"Samples per second: {report['samples_per_second']}")
    if report.get("gpu_peak_memory_allocated_mb") is not None:
        print(f"Peak GPU memory: {report['gpu_peak_memory_allocated_mb']} MB")
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())