"""Classification inference service using the repository's real checkpoint."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from ml.classification.dataset import LABEL_COLUMNS
from ml.classification.pilot import create_model
from ml.classification.preprocessing import normalize_image, resize_image


class ClassificationInputError(ValueError):
    """Raised when an uploaded image cannot be safely processed."""


class ClassificationModelUnavailable(RuntimeError):
    """Raised when the configured classifier checkpoint is unavailable."""


class ClassificationService:
    """Load the classifier lazily and run one-image inference."""

    def __init__(
        self,
        checkpoint_path: Path,
        image_size: tuple[int, int] = (224, 224),
        mean: float = 0.5,
        std: float = 0.5,
        threshold: float = 0.5,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: torch.nn.Module | None = None

    def is_available(self) -> bool:
        """Return whether the configured checkpoint exists."""
        return self.checkpoint_path.is_file()

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        if not self.checkpoint_path.is_file():
            raise ClassificationModelUnavailable(
                f"Classification checkpoint not found: {self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        model = create_model(num_classes=len(LABEL_COLUMNS), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self._model = model
        return model

    def _preprocess(self, payload: bytes) -> torch.Tensor:
        if not payload:
            raise ClassificationInputError("Uploaded image is empty.")
        try:
            with Image.open(BytesIO(payload)) as source:
                image = ImageOps.exif_transpose(source).convert("L")
                image.load()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise ClassificationInputError(
                f"Uploaded file is not a readable image: {type(error).__name__}."
            ) from error

        image = resize_image(image, size=self.image_size)
        image_array = normalize_image(image, mean=self.mean, std=self.std)
        return torch.from_numpy(image_array).unsqueeze(0).to(self.device)

    def predict(self, payload: bytes) -> list[dict[str, object]]:
        """Return finite multi-label predictions for one uploaded image."""
        model = self._load_model()
        image_tensor = self._preprocess(payload)

        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.sigmoid(logits).squeeze(0).float().cpu().numpy()

        if probabilities.shape != (len(LABEL_COLUMNS),):
            raise RuntimeError(
                f"Classifier returned unexpected output shape: {probabilities.shape}."
            )
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Classifier returned non-finite probabilities.")

        return [
            {
                "label": label,
                "probability": float(probability),
                "positive_at_threshold": bool(probability >= self.threshold),
            }
            for label, probability in zip(LABEL_COLUMNS, probabilities, strict=True)
        ]
