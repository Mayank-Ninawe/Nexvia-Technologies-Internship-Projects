"""Train and evaluate the MSD Task09 Spleen 3D U-Net.

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
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import Compose, RandCropByPosNegLabeld

from ml.run_metadata import build_run_metadata, write_run_metadata
from ml.segmentation.model import create_spleen_unet, dice_score, post_process_mask
from ml.segmentation.preprocessing import (
    IMAGE_KEY,
    LABEL_KEY,
    build_preprocessing_transforms,
    create_dataset,
    list_volume_pairs,
)


def build_dynamic_crop_transform(config: dict) -> Compose:
    """Dynamic on-the-fly random positive/negative crop transform."""
    roi_size = config["training"]["roi_size"]

    return Compose(
        [
            RandCropByPosNegLabeld(
                keys=[IMAGE_KEY, LABEL_KEY],
                label_key=LABEL_KEY,
                spatial_size=roi_size,
                pos=2.0,
                neg=1.0,
                num_samples=1,
                image_key=IMAGE_KEY,
                image_threshold=0.0,
            ),
        ]
    )


def build_training_transform(config: dict) -> Compose:
    """Build the dynamic training crop transform."""
    return build_dynamic_crop_transform(config)


def create_training_dataset(config: dict, limit: int | None = None) -> Dataset:
    """Create a decoupled training dataset: deterministic cache + dynamic crops."""
    dataset_config = config["dataset"]

    samples = list_volume_pairs(
        images_dir=Path(dataset_config["images_dir"]),
        labels_dir=Path(dataset_config["labels_dir"]),
    )

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")
        samples = samples[:limit]

    if not samples:
        raise ValueError("No training image/label pairs were found.")

    cached_ds = CacheDataset(
        data=samples,
        transform=build_preprocessing_transforms(config),
        cache_rate=1.0,
        num_workers=config["training"]["num_workers"],
    )
    return Dataset(data=cached_ds, transform=build_dynamic_crop_transform(config))


def train_segmentation(
    config_path: Path,
    output_dir: Path,
    epochs: int,
    train_limit: int | None,
    validation_limit: int | None,
    command: list[str] | None = None,
) -> dict:
    """Train the spleen U-Net and evaluate Dice on whole validation volumes."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    split_config = config["split"]
    train_volume_count = train_limit or int(split_config["train_volume_count"])
    validation_volume_count = validation_limit or int(
        split_config["validation_volume_count"]
    )
    if train_volume_count <= 0 or validation_volume_count <= 0:
        raise ValueError("Training and validation volume counts must be positive.")
    if train_volume_count + validation_volume_count > len(
        list_volume_pairs(
            images_dir=Path(config["dataset"]["images_dir"]),
            labels_dir=Path(config["dataset"]["labels_dir"]),
        )
    ):
        raise ValueError("Training and validation volume counts exceed available data.")
    train_dataset = create_training_dataset(config, limit=train_volume_count)
    validation_dataset = create_dataset(
        config,
        limit=validation_volume_count,
        offset=train_volume_count,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = create_spleen_unet().to(device)
    roi_size = tuple(config["training"]["roi_size"])

    loss_function = DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "best_spleen_unet.pt"
    last_checkpoint = output_dir / "last_spleen_unet.pt"
    log_path = output_dir / "segmentation_training_log.jsonl"

    best_validation_dice = -1.0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in train_loader:
            images = batch[IMAGE_KEY].to(device, non_blocking=True)
            targets = batch[LABEL_KEY].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = loss_function(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu())
            batches += 1

        model.eval()
        validation_dice_values: list[float] = []

        with torch.no_grad():
            for sample in validation_loader:
                images = sample[IMAGE_KEY].to(device)
                targets = sample[LABEL_KEY].to(device)

                with torch.autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda",
                ):
                    logits = sliding_window_inference(
                        inputs=images,
                        roi_size=roi_size,
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.5,
                        mode="gaussian",
                    )

                clean_pred = post_process_mask(
                    logits, threshold=0.5, keep_largest_component=True
                )
                validation_dice_values.append(dice_score(clean_pred, targets))

        mean_validation_dice = (
            sum(validation_dice_values) / len(validation_dice_values)
            if validation_dice_values
            else 0.0
        )
        elapsed = time.perf_counter() - epoch_started

        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "validation_dice": mean_validation_dice,
            "validation_volumes": len(validation_dice_values),
            "elapsed_seconds": round(elapsed, 3),
        }
        history.append(record)

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss {record['train_loss']:.4f} | "
            f"Validation Dice {mean_validation_dice:.4f} | "
            f"Elapsed {elapsed:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "validation_dice": mean_validation_dice,
            "loss": "dice_ce",
            "train_volume_count": len(train_dataset),
            "validation_volume_count": len(validation_dataset),
            "validation_offset": train_volume_count,
            "training_config": config["training"],
            "split_config": split_config,
        }
        torch.save(checkpoint, last_checkpoint)

        if mean_validation_dice > best_validation_dice:
            best_validation_dice = mean_validation_dice
            torch.save(checkpoint, best_checkpoint)
            print(f"New best validation Dice: {best_validation_dice:.4f}")

    run_metadata = build_run_metadata(
        command=command or ["programmatic segmentation training"],
        config_path=config_path,
        checkpoint_path=best_checkpoint,
        checkpoint={
            "epoch": epochs,
            "validation_dice": best_validation_dice,
        },
    )
    write_run_metadata(output_dir / "run_metadata.json", run_metadata)

    return {
        "device": str(device),
        "epochs": epochs,
        "history": history,
        "best_validation_dice": best_validation_dice,
        "best_checkpoint": best_checkpoint.as_posix(),
        "last_checkpoint": last_checkpoint.as_posix(),
        "run_metadata": run_metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/segmentation/config.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/segmentation"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    epochs = args.epochs or int(config["training"]["epochs"])

    try:
        result = train_segmentation(
            config_path=args.config,
            output_dir=args.output_dir,
            epochs=epochs,
            train_limit=args.train_limit,
            validation_limit=args.validation_limit,
            command=sys.argv,
        )
    except (OSError, ValueError, KeyError, RuntimeError, yaml.YAMLError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print("PASS: spleen segmentation training completed.")
    print(f"Device: {result['device']}")
    print(f"Best validation Dice: {result['best_validation_dice']:.4f}")
    print(f"Best checkpoint: {result['best_checkpoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())