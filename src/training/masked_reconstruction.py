"""Patch masking and reconstruction losses for the joint MAE auxiliary task."""

from __future__ import annotations

import torch


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert BxCxHxW images to BxNx(C*P*P) patch vectors."""

    if images.ndim != 4:
        raise ValueError("Expected BxCxHxW images")
    batch, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Image dimensions must be divisible by patch_size")
    grid_h, grid_w = height // patch_size, width // patch_size
    return (
        images.reshape(
            batch, channels, grid_h, patch_size, grid_w, patch_size
        )
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, grid_h * grid_w, channels * patch_size * patch_size)
    )


def unpatchify(
    patches: torch.Tensor,
    *,
    patch_size: int,
    image_height: int,
    image_width: int,
    channels: int = 3,
) -> torch.Tensor:
    """Reassemble BxNx(C*P*P) patch vectors into BxCxHxW images."""

    batch = patches.shape[0]
    grid_h, grid_w = image_height // patch_size, image_width // patch_size
    expected = (grid_h * grid_w, channels * patch_size * patch_size)
    if tuple(patches.shape[1:]) != expected:
        raise ValueError(f"Patch tensor has shape {tuple(patches.shape[1:])}, expected {expected}")
    return (
        patches.reshape(
            batch, grid_h, grid_w, channels, patch_size, patch_size
        )
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels, image_height, image_width)
    )


def mask_images(
    images: torch.Tensor,
    *,
    patch_size: int,
    mask_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask random patches and return masked images, normalized targets, mask."""

    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be strictly between zero and one")
    patches = patchify(images, patch_size)
    patch_count = patches.shape[1]
    masked_count = max(1, min(patch_count - 1, round(mask_ratio * patch_count)))
    noise = torch.rand(
        patches.shape[0], patch_count, device=patches.device
    )
    masked_indices = noise.topk(masked_count, dim=1, largest=False).indices
    mask = torch.zeros(
        patches.shape[0], patch_count, dtype=torch.bool, device=patches.device
    )
    mask.scatter_(1, masked_indices, True)

    masked_patches = patches.masked_fill(mask.unsqueeze(-1), 0.0)
    masked_images = unpatchify(
        masked_patches,
        patch_size=patch_size,
        image_height=images.shape[2],
        image_width=images.shape[3],
        channels=images.shape[1],
    )

    target_mean = patches.mean(dim=-1, keepdim=True)
    target_variance = patches.var(dim=-1, keepdim=True, unbiased=False)
    normalized_targets = (patches - target_mean) / torch.sqrt(
        target_variance + 1e-6
    )
    return masked_images, normalized_targets, mask


def masked_patch_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over masked patches only."""

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction shape {tuple(predictions.shape)} does not match "
            f"target shape {tuple(targets.shape)}"
        )
    per_patch = (predictions.float() - targets.float()).square().mean(dim=-1)
    return per_patch.masked_select(mask).mean()
