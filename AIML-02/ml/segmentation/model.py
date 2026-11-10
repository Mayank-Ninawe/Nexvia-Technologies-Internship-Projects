"""3D U-Net model and segmentation metrics for MSD Task09 Spleen."""

from __future__ import annotations

import torch
from monai.networks.layers import Norm
from monai.networks.nets import UNet


def create_spleen_unet(prior_prob: float = 0.01) -> UNet:
    """Create a compact 3D U-Net suitable for a 6 GB GPU pilot.

    Parameters
    ----------
    prior_prob : float
        Foreground class prior probability for final convolution bias initialization.
        Setting to ~0.01 prevents the initial sigmoid explosion and catastrophic
        over-segmentation on background voxels. Set to 0.0 to disable.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.INSTANCE,
    )

    if prior_prob > 0.0:
        import math

        bias_val = -math.log((1.0 - prior_prob) / prior_prob)
        for mod in model.modules():
            if (
                isinstance(mod, torch.nn.Conv3d)
                and mod.out_channels == 1
                and mod.bias is not None
            ):
                mod.bias.data.fill_(bias_val)

    return model


def dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    """Compute binary Dice score for a single-channel prediction batch."""
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction shape {tuple(predictions.shape)} does not match "
            f"target shape {tuple(targets.shape)}."
        )
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("Predictions and targets must contain only finite values.")

    unique_preds = torch.unique(predictions)
    if torch.isin(unique_preds, torch.tensor([0.0, 1.0], device=predictions.device)).all():
        binary_predictions = predictions.float()
    else:
        binary_predictions = (torch.sigmoid(predictions) >= threshold).float()

    targets = targets.float()
    unique_targets = torch.unique(targets)
    if not torch.isin(unique_targets, torch.tensor([0.0, 1.0], device=targets.device)).all():
        raise ValueError("Targets must contain only binary values.")

    intersection = (binary_predictions * targets).sum()
    denominator = binary_predictions.sum() + targets.sum()

    if denominator == 0:
        return 1.0

    return float((2.0 * intersection + epsilon) / (denominator + epsilon))


def post_process_mask(
    predictions: torch.Tensor,
    threshold: float = 0.5,
    keep_largest_component: bool = True,
) -> torch.Tensor:
    """Binarize predictions and optionally retain only the largest connected component.

    Parameters
    ----------
    predictions : torch.Tensor
        Model output (logits or probabilities) of shape (B, 1, D, H, W) or (1, D, H, W).
    threshold : float
        Probability threshold for binarization (default 0.5).
    keep_largest_component : bool
        Whether to retain only the largest connected component.

    Returns
    -------
    torch.Tensor
        Cleaned binary tensor with identical shape and dtype float32.
    """
    if predictions.ndim not in (4, 5):
        raise ValueError(
            f"Expected predictions to have 4 or 5 dimensions, got shape {tuple(predictions.shape)}."
        )

    probs = predictions
    if probs.min() < 0.0 or probs.max() > 1.0:
        probs = torch.sigmoid(probs)

    binary_mask = (probs >= threshold).float()
    if not keep_largest_component or binary_mask.sum() == 0:
        return binary_mask

    import numpy as np
    import scipy.ndimage as ndi

    def _clean_single_volume(vol: torch.Tensor) -> torch.Tensor:
        # vol shape: (1, D, H, W)
        mask_np = vol[0].detach().cpu().numpy().astype(bool)
        if not mask_np.any():
            return vol

        labeled, num_features = ndi.label(mask_np)
        if num_features <= 1:
            return vol

        counts = np.bincount(labeled.flat)
        counts[0] = 0
        largest_label = counts.argmax()
        largest_mask = (labeled == largest_label)

        cleaned_vol = torch.from_numpy(largest_mask).unsqueeze(0).to(
            dtype=vol.dtype, device=vol.device
        )
        return cleaned_vol

    if binary_mask.ndim == 5:
        cleaned_batches = [_clean_single_volume(b) for b in binary_mask]
        return torch.stack(cleaned_batches)
    return _clean_single_volume(binary_mask)
