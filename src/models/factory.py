"""Model factory with an explicit no-external-pretraining policy."""

from __future__ import annotations

from functools import partial

import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.vision_transformer import VisionTransformer


def build_model(
    name: str,
    *,
    num_classes: int,
    image_size: int,
) -> nn.Module:
    """Construct a randomly initialized supervised control model."""

    if name == "resnet18":
        # weights=None is intentional and required by the assignment.
        model = resnet18(weights=None, zero_init_residual=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "vit_tiny":
        if image_size % 16:
            raise ValueError("vit_tiny image_size must be divisible by patch size 16")
        return VisionTransformer(
            image_size=image_size,
            patch_size=16,
            num_layers=6,
            num_heads=3,
            hidden_dim=192,
            mlp_dim=768,
            dropout=0.10,
            attention_dropout=0.0,
            num_classes=num_classes,
            representation_size=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

    raise ValueError(f"Unknown model {name!r}; expected 'resnet18' or 'vit_tiny'")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable

