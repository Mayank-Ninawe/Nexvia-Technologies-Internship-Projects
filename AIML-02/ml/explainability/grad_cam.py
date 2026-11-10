"""Grad-CAM attribution for the NIH DenseNet-121 classifier.

Grad-CAM is a model-dependent visualization. It does not prove that a model is
clinically correct and must not be interpreted as a diagnosis.

Educational use only; not for clinical use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn

from ml.classification.dataset import LABEL_COLUMNS
from ml.classification.pilot import create_model
from ml.classification.preprocessing import load_grayscale_image, resize_image


class GradCAM:
    """Generate Grad-CAM heatmaps for a supported DenseNet classifier."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._forward_hook = target_layer.register_forward_hook(
            self._save_activation
        )
        self._backward_hook = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(
        self,
        _module: nn.Module,
        _inputs: tuple,
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, class_index: int) -> np.ndarray:
        """Return a normalized heatmap with shape (height, width)."""
        if self.activations is not None:
            self.activations = None
        if self.gradients is not None:
            self.gradients = None

        self.model.eval()
        input_tensor = image_tensor.unsqueeze(0).requires_grad_(True)

        logits = self.model(input_tensor)
        if class_index < 0 or class_index >= logits.shape[1]:
            raise ValueError(
                f"class_index must be between 0 and {logits.shape[1] - 1}."
            )

        self.model.zero_grad(set_to_none=True)
        logits[0, class_index].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture required values.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        weighted_activations = weights * self.activations
        heatmap = weighted_activations.sum(dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)

        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        if heatmap_max - heatmap_min == 0:
            heatmap = torch.zeros_like(heatmap)
        else:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

        return heatmap.squeeze().cpu().numpy()

    def remove_hooks(self) -> None:
        self._forward_hook.remove()
        self._backward_hook.remove()


def create_overlay(
    image_path: Path,
    heatmap: np.ndarray,
    output_path: Path,
    size: tuple[int, int] = (224, 224),
    alpha: float = 0.45,
) -> Path:
    """Create an aligned original-plus-heatmap overlay PNG."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    original = resize_image(load_grayscale_image(image_path), size=size)
    original_rgb = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0

    heatmap_resized = np.asarray(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            size, resample=Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0

    colormap = matplotlib.colormaps["jet"]
    colored_heatmap = colormap(heatmap_resized)[..., :3]

    overlay = alpha * colored_heatmap + (1.0 - alpha) * original_rgb
    overlay = np.clip(overlay, 0.0, 1.0)
    overlay_image = Image.fromarray((overlay * 255).astype(np.uint8))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_image.save(output_path)
    return output_path


def generate_grad_cam(
    config_path: Path,
    checkpoint_path: Path,
    image_path: Path,
    label: str,
    output_path: Path,
) -> dict:
    """Generate Grad-CAM and prediction metadata for one NIH image."""
    if label not in LABEL_COLUMNS:
        raise ValueError(
            f"Unsupported label {label!r}. Supported labels: {LABEL_COLUMNS}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    image_config = config["image"]
    size = (image_config["resize_width"], image_config["resize_height"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    from ml.classification.preprocessing import normalize_image

    image = resize_image(load_grayscale_image(image_path), size=size)
    image_array = normalize_image(
        image,
        mean=image_config["normalize_mean"],
        std=image_config["normalize_std"],
    )
    image_tensor = torch.from_numpy(image_array).to(device)

    grad_cam = GradCAM(model, model.features.denseblock4)
    try:
        class_index = LABEL_COLUMNS.index(label)
        heatmap = grad_cam.generate(image_tensor, class_index)

        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0))
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        predicted_index = int(probabilities.argmax())
        overlay_path = create_overlay(image_path, heatmap, output_path, size=size)
    finally:
        grad_cam.remove_hooks()

    result = {
        "image_path": image_path.as_posix(),
        "target_label": label,
        "target_probability": round(float(probabilities[class_index]), 6),
        "predicted_label": LABEL_COLUMNS[predicted_index],
        "predicted_probability": round(float(probabilities[predicted_index]), 6),
        "heatmap_shape": list(heatmap.shape),
        "heatmap_min": round(float(heatmap.min()), 6),
        "heatmap_max": round(float(heatmap.max()), 6),
        "overlay_path": overlay_path.as_posix(),
        "device": str(device),
        "interpretation_limitation": (
            "Grad-CAM is a model-dependent attribution visualization; "
            "it is not clinical evidence or a diagnosis."
        ),
    }

    sidecar_path = output_path.with_suffix(".json")
    with sidecar_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    return result


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
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/grad_cam_overlay.png"),
    )
    args = parser.parse_args()

    try:
        result = generate_grad_cam(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            image_path=args.image,
            label=args.label,
            output_path=args.output,
        )
    except (OSError, ValueError, KeyError, RuntimeError, yaml.YAMLError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print("PASS: Grad-CAM generated.")
    print(f"Target label: {result['target_label']}")
    print(f"Target probability: {result['target_probability']}")
    print(f"Predicted label: {result['predicted_label']}")
    print(f"Overlay: {result['overlay_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())