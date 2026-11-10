"""Tests for segmentation evaluation reporting."""

import pytest
import torch

from ml.segmentation.model import dice_score


def test_dice_score_handles_nonempty_prediction_and_target():
    targets = torch.zeros(1, 1, 8, 8, 8)
    targets[:, :, 2:5, 2:5, 2:5] = 1.0

    logits = targets * 10.0 - 5.0

    score = dice_score(logits, targets)

    assert 0.0 <= score <= 1.0
    assert score > 0.9


def test_dice_score_rejects_non_binary_targets():
    logits = torch.zeros(1, 1, 8, 8, 8)
    targets = torch.full((1, 1, 8, 8, 8), 2.0)

    with pytest.raises(ValueError, match="binary"):
        dice_score(logits, targets)