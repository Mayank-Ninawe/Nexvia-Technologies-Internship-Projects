"""Time-budgeted classifier fine-tuning using validation-only selection.

Educational use only; not for clinical use.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ml.classification.dataset import LABEL_COLUMNS, ChestXrayDataset
from ml.classification.evaluate import collect_predictions, compute_metrics, load_model
from ml.classification.validate_checkpoint import sha256_file


class EpochAugmentedDataset(ChestXrayDataset):
    """Apply reproducible horizontal flips with epoch-dependent decisions."""

    def __init__(self, *args, seed: int = 42, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.epoch = 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, target = super().__getitem__(index)
        seed = self.seed + self.epoch * len(self) + index
        if random.Random(seed).random() < 0.5:
            image = torch.flip(image, dims=(-1,))
        return image, target


def write_json(path: Path, content: dict) -> None:
    path.write_text(
        json.dumps(content, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def atomic_save(content: dict, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(content, temporary)
    temporary.replace(path)


def validation_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    logits, targets = collect_predictions(model, loader, device)
    metrics = compute_metrics(logits, targets, threshold=0.5)
    metrics["validation_samples"] = metrics.pop("test_samples")
    metrics["split"] = "validation"
    return metrics


def finetune(
    config_path: Path,
    checkpoint_path: Path,
    baseline_path: Path,
    output_dir: Path,
    epochs: int = 6,
    hours: float = 5.0,
    learning_rate: float = 2e-5,
    patience: int = 2,
    seed: int = 42,
) -> dict:
    """Fine-tune without modifying the original checkpoint or using test data."""
    if epochs < 1 or hours <= 0 or learning_rate <= 0 or patience < 1:
        raise ValueError("Epochs, hours, learning rate, and patience must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this planned GPU experiment.")
    if output_dir.exists():
        raise FileExistsError(f"Use a new output directory: {output_dir}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["dataset"]["manifest_file"]).with_name(
        "nih_chestxray14_patient_splits.csv"
    )

    checkpoint_hash = sha256_file(checkpoint_path)
    manifest_hash = sha256_file(manifest_path)
    config_hash = sha256_file(config_path)

    expected = {
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "config_sha256": config_hash,
        "split": "validation",
    }
    for key, value in expected.items():
        if baseline.get(key) != value:
            raise ValueError(f"Baseline mismatch for {key}; do not mix artifacts.")

    if baseline.get("label_columns") != list(LABEL_COLUMNS):
        raise ValueError("Baseline label order does not match current code.")

    best_ap = baseline.get("macro_average_precision")
    if best_ap is None or not math.isfinite(float(best_ap)):
        raise ValueError("Baseline macro average precision is undefined.")
    best_ap = float(best_ap)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")
    model, source_checkpoint = load_model(checkpoint_path, device)
    stored_labels = source_checkpoint.get("label_columns")
    if stored_labels is not None and list(stored_labels) != list(LABEL_COLUMNS):
        raise ValueError("Checkpoint label order mismatch.")
    del source_checkpoint

    image_config = config["image"]
    settings = {
        "manifest_path": manifest_path,
        "image_size": (
            image_config["resize_width"],
            image_config["resize_height"],
        ),
        "mean": image_config["normalize_mean"],
        "std": image_config["normalize_std"],
        "augment": False,
    }
    train_dataset = EpochAugmentedDataset(
        split="train",
        seed=seed,
        **settings,
    )
    validation_dataset = ChestXrayDataset(split="validation", **settings)

    if len(validation_dataset) != baseline.get("validation_samples"):
        raise ValueError("Validation sample count differs from baseline.")

    workers = int(config["training"]["num_workers"])
    generator = torch.Generator().manual_seed(seed)
    loader_settings = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_settings,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_settings,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "finetune_source.py").write_text(
        Path(__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    metadata = {
        "command": sys.argv,
        "git_revision": revision,
        "git_status_porcelain": dirty_status,
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "config_snapshot": config,
        "source_checkpoint": checkpoint_path.as_posix(),
        "source_checkpoint_sha256": checkpoint_hash,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": manifest_hash,
        "config_sha256": config_hash,
        "baseline_report_sha256": sha256_file(baseline_path),
        "seed": seed,
        "additional_epochs_limit": epochs,
        "hours_budget": hours,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "patience": patience,
        "selection_metric": "validation_macro_average_precision",
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "implementation_hashes": {
            path: sha256_file(Path(path))
            for path in (
                "ml/classification/finetune.py",
                "ml/classification/dataset.py",
                "ml/classification/preprocessing.py",
                "ml/classification/pilot.py",
                "ml/classification/evaluate.py",
            )
        },
    }
    write_json(output_dir / "run_config.json", metadata)
    write_json(output_dir / "baseline_validation.json", baseline)

    best_path = output_dir / "best_classifier.pt"
    atomic_save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "label_columns": list(LABEL_COLUMNS),
            "validation_macro_average_precision": best_ap,
            "origin": "unchanged_baseline",
            "source_checkpoint_sha256": checkpoint_hash,
        },
        best_path,
    )

    started = time.perf_counter()
    deadline = started + hours * 3600
    previous_epoch_seconds = None
    best_epoch = 0
    non_improving = 0
    completed = 0
    stop_reason = "epoch_limit"

    print(
        f"CUDA: {metadata['gpu']} | Train: {len(train_dataset)} | "
        f"Validation: {len(validation_dataset)} | Baseline AP: {best_ap:.6f}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        remaining = deadline - time.perf_counter()
        if remaining <= 0 or (
            previous_epoch_seconds is not None
            and remaining < previous_epoch_seconds * 1.15
        ):
            stop_reason = "insufficient_budget_for_another_epoch"
            break

        epoch_started = time.perf_counter()
        train_dataset.epoch = epoch
        model.train()
        torch.cuda.reset_peak_memory_stats()
        loss_sum = 0.0
        samples = 0
        interrupted_for_budget = False

        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            if time.perf_counter() >= deadline:
                interrupted_for_budget = True
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, targets)

            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Non-finite training loss; stopping experiment.")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.shape[0]
            loss_sum += loss.item() * batch_size
            samples += batch_size

            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Batch {batch_index}/{len(train_loader)} | "
                    f"Mean loss {loss_sum / samples:.5f}",
                    flush=True,
                )

        if interrupted_for_budget:
            stop_reason = "budget_reached_mid_epoch_unvalidated_weights_discarded"
            break

        metrics = validation_metrics(model, validation_loader, device)
        ap = metrics["macro_average_precision"]
        if ap is None or not math.isfinite(float(ap)):
            raise RuntimeError("Validation macro average precision is undefined.")

        completed = epoch
        improved = float(ap) > best_ap
        if improved:
            best_ap = float(ap)
            best_epoch = epoch
            non_improving = 0
        else:
            non_improving += 1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "label_columns": list(LABEL_COLUMNS),
            "validation_macro_average_precision": float(ap),
            "best_validation_macro_average_precision": best_ap,
            "source_checkpoint_sha256": checkpoint_hash,
            "origin": "finetuned_candidate",
        }
        atomic_save(checkpoint, output_dir / "last_classifier.pt")
        if improved:
            atomic_save(checkpoint, best_path)
            write_json(output_dir / "best_validation.json", metrics)

        previous_epoch_seconds = time.perf_counter() - epoch_started
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / samples,
            "validation_macro_average_precision": float(ap),
            "validation_macro_auroc": metrics["macro_auroc"],
            "improved_over_best": improved,
            "elapsed_seconds": round(previous_epoch_seconds, 3),
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated() / 1024**2, 2
            ),
        }
        write_json(output_dir / f"validation_epoch_{epoch:02d}.json", metrics)
        with (output_dir / "training_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)

        if non_improving >= patience:
            stop_reason = "early_stopping"
            break

    summary = {
        "completed_additional_epochs": completed,
        "best_additional_epoch": best_epoch,
        "baseline_validation_macro_ap": baseline["macro_average_precision"],
        "best_validation_macro_ap": best_ap,
        "improved_over_baseline": best_epoch > 0,
        "best_checkpoint": best_path.as_posix(),
        "best_checkpoint_sha256": sha256_file(best_path),
        "stop_reason": stop_reason,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/classification/config.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--hours", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    finetune(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        baseline_path=args.baseline,
        output_dir=args.output_dir,
        epochs=args.epochs,
        hours=args.hours,
        learning_rate=args.lr,
        patience=args.patience,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())