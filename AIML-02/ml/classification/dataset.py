"""PyTorch dataset for the NIH ChestX-ray14 classification workflow."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.classification.preprocessing import preprocess_image

LABEL_COLUMNS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


class ChestXrayDataset(Dataset):
    """Load images and multi-label targets from a patient-level split manifest."""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        image_size: tuple[int, int] = (224, 224),
        mean: float = 0.5,
        std: float = 0.5,
        augment: bool = False,
        limit: int | None = None,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(
                "split must be one of: 'train', 'validation', or 'test'."
            )

        self.manifest_path = Path(manifest_path)
        self.split = split
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = augment

        self.rows = self._read_rows(limit)
        if not self.rows:
            raise ValueError(
                f"No rows found for split {split!r} in {self.manifest_path}."
            )

    def _read_rows(self, limit: int | None) -> list[dict[str, str]]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {self.manifest_path}")

        rows: list[dict[str, str]] = []

        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"image_path", "split", *LABEL_COLUMNS}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Manifest is missing required columns: {sorted(missing)}"
                )

            for row in reader:
                if row["split"] != self.split:
                    continue
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break

        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image_path = Path(row["image_path"])

        image_array = preprocess_image(
            image_path,
            size=self.image_size,
            mean=self.mean,
            std=self.std,
            augment=self.augment,
            random_seed=index,
        )

        image_tensor = torch.from_numpy(image_array)

        labels = np.asarray(
            [float(row[column]) for column in LABEL_COLUMNS],
            dtype=np.float32,
        )
        label_tensor = torch.from_numpy(labels)

        return image_tensor, label_tensor