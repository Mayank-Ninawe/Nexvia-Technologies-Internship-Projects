"""Grad-CAM service for supported classification inputs."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from ml.explainability.grad_cam import generate_grad_cam


class ExplainabilityInputError(ValueError):
    """Raised when a Grad-CAM request is invalid."""


class ExplainabilityService:
    """Generate controlled, in-memory Grad-CAM responses."""

    def __init__(self, config_path: Path, checkpoint_path: Path) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

    def generate(self, payload: bytes, label: str) -> dict[str, object]:
        if not payload:
            raise ExplainabilityInputError("Uploaded image is empty.")

        with tempfile.TemporaryDirectory(prefix="medical-imaging-gradcam-") as directory:
            directory_path = Path(directory)
            image_path = directory_path / "input-image"
            output_path = directory_path / "overlay.png"
            image_path.write_bytes(payload)

            try:
                result = generate_grad_cam(
                    config_path=self.config_path,
                    checkpoint_path=self.checkpoint_path,
                    image_path=image_path,
                    label=label,
                    output_path=output_path,
                )
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                raise ExplainabilityInputError(str(error)) from error

            overlay_base64 = base64.b64encode(output_path.read_bytes()).decode("ascii")

        result.pop("image_path", None)
        result.pop("overlay_path", None)
        result["overlay_base64"] = overlay_base64
        return result
