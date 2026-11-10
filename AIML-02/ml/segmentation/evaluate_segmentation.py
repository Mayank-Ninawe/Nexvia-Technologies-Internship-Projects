"""Evaluate the trained spleen U-Net on validation volumes.

Educational use only; not for clinical use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from monai.data import DataLoader
from monai.inferers import sliding_window_inference

from ml.run_metadata import build_run_metadata
from ml.segmentation.model import create_spleen_unet, dice_score, post_process_mask
from ml.segmentation.preprocessing import IMAGE_KEY, LABEL_KEY, create_dataset


def evaluate_segmentation(
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    validation_limit: int | None,
    command: list[str] | None = None,
) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = create_spleen_unet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    validation_offset = int(
        checkpoint.get("validation_offset", config["split"]["train_volume_count"])
    )
    dataset = create_dataset(
        config,
        limit=validation_limit,
        offset=validation_offset,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    roi_size = tuple(config["training"]["roi_size"])
    results = []

    with torch.no_grad():
        for index, sample in enumerate(loader, start=1):
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

            cleaned_pred = post_process_mask(logits, threshold=0.5, keep_largest_component=True)
            dice = dice_score(cleaned_pred, targets)
            raw_dice = dice_score(logits, targets)
            target_pixels = int(targets.sum().item())
            raw_predicted_pixels = int((torch.sigmoid(logits) >= 0.5).sum().item())
            predicted_pixels = int(cleaned_pred.sum().item())

            results.append(
                {
                    "volume_index": index,
                    "dice": round(dice, 6),
                    "raw_dice": round(raw_dice, 6),
                    "target_spleen_pixels": target_pixels,
                    "predicted_spleen_pixels": predicted_pixels,
                    "raw_predicted_spleen_pixels": raw_predicted_pixels,
                    "image_shape": list(images.shape),
                }
            )

            print(
                f"Volume {index}: Dice={dice:.4f} (Raw={raw_dice:.4f}), "
                f"target pixels={target_pixels}, "
                f"cleaned pixels={predicted_pixels} (raw={raw_predicted_pixels})"
            )

    dice_values = [result["dice"] for result in results]
    report = {
        "checkpoint_path": checkpoint_path.as_posix(),
        "device": str(device),
        "run_metadata": build_run_metadata(
            command=command or ["programmatic segmentation evaluation"],
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
        ),
        "validation_volumes": len(results),
        "mean_dice": round(sum(dice_values) / len(dice_values), 6)
        if dice_values
        else None,
        "per_volume": results,
        "limitation": (
            "This is a bounded pilot evaluation on a small validation subset; "
            "it does not represent full MSD Task09 test-set performance."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/segmentation/config.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/segmentation/best_spleen_unet.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/spleen_segmentation_metrics.json"),
    )
    parser.add_argument("--validation-limit", type=int, default=2)
    args = parser.parse_args()

    try:
        report = evaluate_segmentation(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            validation_limit=args.validation_limit,
            command=sys.argv,
        )
    except (OSError, ValueError, KeyError, RuntimeError, yaml.YAMLError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print("PASS: spleen segmentation evaluation completed.")
    print(f"Validation volumes: {report['validation_volumes']}")
    print(f"Mean Dice: {report['mean_dice']}")
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())