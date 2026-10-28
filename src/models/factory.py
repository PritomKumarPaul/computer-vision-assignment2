"""Model factory with an explicit no-external-pretraining policy."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.vision_transformer import VisionTransformer


class ResNet18Encoder(nn.Module):
    feature_dim = 512

    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None, zero_init_residual=True)
        self.backbone.fc = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


class TinyViTEncoder(nn.Module):
    feature_dim = 192

    def __init__(self, image_size: int) -> None:
        super().__init__()
        if image_size % 16:
            raise ValueError("vit_tiny image_size must be divisible by patch size 16")
        self.backbone = VisionTransformer(
            image_size=image_size,
            patch_size=16,
            num_layers=6,
            num_heads=3,
            hidden_dim=self.feature_dim,
            mlp_dim=768,
            dropout=0.10,
            attention_dropout=0.0,
            num_classes=1,
            representation_size=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.backbone.heads = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


class EncoderClassifier(nn.Module):
    """Classifier used for independent SSL probing and fine-tuning runs."""

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        num_classes: int,
        *,
        freeze_encoder: bool,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> "EncoderClassifier":
        super().train(mode)
        if self.freeze_encoder:
            # A linear probe must not update BatchNorm running statistics.
            self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(images)
        else:
            features = self.encoder(images)
        return self.classifier(features)


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


def build_encoder(name: str, *, image_size: int) -> tuple[nn.Module, int]:
    """Construct a randomly initialized encoder for SSL."""

    if name == "resnet18":
        encoder = ResNet18Encoder()
    elif name == "vit_tiny":
        encoder = TinyViTEncoder(image_size)
    else:
        raise ValueError(f"Unknown encoder {name!r}")
    return encoder, encoder.feature_dim


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable

