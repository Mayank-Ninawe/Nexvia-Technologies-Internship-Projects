"""Train the NIH ChestX-ray14 multi-label classifier.

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

from ml.classification.dataset import LABEL_COLUMNS, ChestXrayDataset
from ml.classification.pilot import create_model
from ml.run_metadata import build_run_metadata, write_run_metadata


def build_loaders(
    config: dict,
    train_limit: int | None,
    validation_limit: int | None,
) -> tuple[DataLoader, DataLoader]:
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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Return mean validation loss."""
    model.eval()
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            total_loss += float(loss.detach().cpu())
            batches += 1

    return total_loss / max(batches, 1)


def train(
    config_path: Path,
    output_dir: Path,
    epochs: int | None,
    train_limit: int | None,
    validation_limit: int | None,
    resume_checkpoint: Path | None,
    command: list[str] | None = None,
) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    training_config = config["training"]
    total_epochs = epochs if epochs is not None else int(training_config["epochs"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, validation_loader = build_loaders(
        config,
        train_limit=train_limit,
        validation_limit=validation_limit,
    )

    model = create_model(num_classes=len(LABEL_COLUMNS), pretrained=True)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch = 1
    best_validation_loss = float("inf")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.jsonl"
    best_checkpoint_path = output_dir / "best_classifier.pt"
    last_checkpoint_path = output_dir / "last_classifier.pt"

    if resume_checkpoint is not None:
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_checkpoint}"
            )

        checkpoint = torch.load(
            resume_checkpoint,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])

        print(
            f"Resumed from epoch {checkpoint['epoch']} with "
            f"best validation loss {best_validation_loss:.4f}."
        )

    history: list[dict] = []

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_total = 0.0
        train_batches = 0

        for batch_index, (images, targets) in enumerate(train_loader, start=1):
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

            if batch_index % 50 == 0:
                print(
                    f"Epoch {epoch}/{total_epochs} | "
                    f"Batch {batch_index}/{len(train_loader)} | "
                    f"Loss {loss.detach().cpu():.4f}"
                )

        validation_loss = evaluate(model, validation_loader, criterion, device)
        elapsed = time.perf_counter() - epoch_started

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss_total / max(train_batches, 1),
            "validation_loss": validation_loss,
            "elapsed_seconds": round(elapsed, 3),
            "train_samples": len(train_loader.dataset),
            "validation_samples": len(validation_loader.dataset),
        }
        history.append(epoch_record)

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record) + "\n")

        print(
            f"Epoch {epoch}/{total_epochs} complete | "
            f"Train loss {epoch_record['train_loss']:.4f} | "
            f"Validation loss {validation_loss:.4f} | "
            f"Elapsed {elapsed:.1f}s"
        )

        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_validation_loss": best_validation_loss,
            "validation_loss": validation_loss,
            "label_columns": LABEL_COLUMNS,
        }
        torch.save(last_checkpoint, last_checkpoint_path)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            last_checkpoint["best_validation_loss"] = best_validation_loss
            torch.save(last_checkpoint, best_checkpoint_path)
            print(f"New best validation loss: {best_validation_loss:.4f}")

    run_metadata = build_run_metadata(
        command=command or ["programmatic training"],
        config_path=config_path,
        manifest_path=Path(config["dataset"]["manifest_file"]).with_name(
            "nih_chestxray14_patient_splits.csv"
        ),
        checkpoint_path=best_checkpoint_path,
        checkpoint={
            "epoch": total_epochs,
            "validation_loss": validation_loss,
            "best_validation_loss": best_validation_loss,
        },
    )
    write_run_metadata(output_dir / "run_metadata.json", run_metadata)

    return {
        "device": str(device),
        "epochs_completed": total_epochs,
        "history": history,
        "best_validation_loss": best_validation_loss,
        "best_checkpoint": best_checkpoint_path.as_posix(),
        "last_checkpoint": last_checkpoint_path.as_posix(),
        "run_metadata": run_metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/classification/config.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/classification"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to last_classifier.pt from an interrupted run.",
    )
    args = parser.parse_args()

    try:
        result = train(
            config_path=args.config,
            output_dir=args.output_dir,
            epochs=args.epochs,
            train_limit=args.train_limit,
            validation_limit=args.validation_limit,
            resume_checkpoint=args.resume,
            command=sys.argv,
        )
    except (OSError, ValueError, KeyError, RuntimeError, yaml.YAMLError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print("PASS: classifier training completed.")
    print(f"Best validation loss: {result['best_validation_loss']:.4f}")
    print(f"Best checkpoint: {result['best_checkpoint']}")
    print(f"Last checkpoint: {result['last_checkpoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())