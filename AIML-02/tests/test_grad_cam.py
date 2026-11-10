"""Tests for DenseNet Grad-CAM generation."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ml.classification.dataset import LABEL_COLUMNS
from ml.classification.pilot import create_model
from ml.explainability.grad_cam import GradCAM, create_overlay


@pytest.fixture
def model():
    return create_model(num_classes=len(LABEL_COLUMNS), pretrained=False)


def test_grad_cam_heatmap_shape_and_range(model):
    grad_cam = GradCAM(model, model.features.denseblock4)
    image = torch.randn(1, 224, 224)

    try:
        heatmap = grad_cam.generate(image, class_index=1)
    finally:
        grad_cam.remove_hooks()

    assert heatmap.shape == (7, 7)
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_grad_cam_rejects_invalid_class_index(model):
    grad_cam = GradCAM(model, model.features.denseblock4)
    image = torch.randn(1, 224, 224)

    try:
        with pytest.raises(ValueError, match="class_index"):
            grad_cam.generate(image, class_index=len(LABEL_COLUMNS))
    finally:
        grad_cam.remove_hooks()


def test_create_overlay_writes_aligned_png(tmp_path: Path):
    image_path = tmp_path / "image.png"
    Image.new("L", (64, 64), color=128).save(image_path)

    heatmap = np.random.rand(7, 7).astype(np.float32)
    output = tmp_path / "overlay.png"

    result = create_overlay(
        image_path,
        heatmap,
        output,
        size=(224, 224),
        alpha=0.45,
    )

    assert result == output
    with Image.open(output) as overlay:
        assert overlay.size == (224, 224)
        assert overlay.mode == "RGB"


def test_create_overlay_rejects_invalid_alpha(tmp_path: Path):
    image_path = tmp_path / "image.png"
    Image.new("L", (16, 16), color=128).save(image_path)

    with pytest.raises(ValueError, match="alpha"):
        create_overlay(
            image_path,
            np.zeros((7, 7), dtype=np.float32),
            tmp_path / "overlay.png",
            alpha=1.5,
        )