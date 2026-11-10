"""Tests for the 3D spleen U-Net and Dice metric."""

import pytest
import torch

from ml.segmentation.model import create_spleen_unet, dice_score, post_process_mask


def test_create_spleen_unet_output_shape():
    model = create_spleen_unet()
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(1, 1, 32, 32, 32))

    assert output.shape == (1, 1, 32, 32, 32)
    assert torch.isfinite(output).all()


def test_post_process_mask_keeps_largest_component():
    mask = torch.full((1, 1, 16, 16, 16), -10.0)
    # Small isolated island of 2x2x2 = 8 voxels
    mask[0, 0, 1:3, 1:3, 1:3] = 10.0
    # Large primary organ of 4x4x4 = 64 voxels
    mask[0, 0, 8:12, 8:12, 8:12] = 10.0

    cleaned = post_process_mask(mask, threshold=0.5, keep_largest_component=True)

    assert cleaned.shape == mask.shape
    # Small island must be removed
    assert cleaned[0, 0, 1:3, 1:3, 1:3].sum() == 0
    # Large component must be preserved
    assert cleaned[0, 0, 8:12, 8:12, 8:12].sum() == 64



def test_dice_score_perfect_prediction():
    targets = torch.zeros(1, 1, 8, 8, 8)
    targets[:, :, 2:5, 2:5, 2:5] = 1.0

    logits = torch.where(
        targets.bool(),
        torch.tensor(10.0),
        torch.tensor(-10.0),
    )

    assert dice_score(logits, targets) == 1.0


def test_dice_score_empty_prediction_and_target():
    logits = torch.full((1, 1, 8, 8, 8), -10.0)
    targets = torch.zeros(1, 1, 8, 8, 8)

    assert dice_score(logits, targets) == 1.0


def test_dice_score_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        dice_score(
            torch.randn(1, 1, 8, 8, 8),
            torch.zeros(1, 1, 8, 8, 4),
        )