"""DINO-style EMA self-distillation trained entirely from scratch."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


class DINOHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, output_dim, bias=False)
        )
        with torch.no_grad():
            self.last.parametrizations.weight.original0.fill_(1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.last(F.normalize(self.mlp(features), dim=-1))


class DINO(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        *,
        hidden_dim: int,
        bottleneck_dim: int,
        output_dim: int,
        student_temperature: float,
        center_momentum: float,
    ) -> None:
        super().__init__()
        self.student_encoder = encoder
        self.student_head = DINOHead(feature_dim, hidden_dim, bottleneck_dim, output_dim)
        self.teacher_encoder = copy.deepcopy(encoder)
        # Build an identical teacher head and copy its state.  This avoids
        # relying on deepcopy behavior for weight-normalized parameters.
        self.teacher_head = DINOHead(
            feature_dim, hidden_dim, bottleneck_dim, output_dim
        )
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for parameter in list(self.teacher_encoder.parameters()) + list(
            self.teacher_head.parameters()
        ):
            parameter.requires_grad = False
        self.student_temperature = student_temperature
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, output_dim))

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for student, teacher in zip(
            list(self.student_encoder.parameters()) + list(self.student_head.parameters()),
            list(self.teacher_encoder.parameters()) + list(self.teacher_head.parameters()),
        ):
            teacher.data.mul_(momentum).add_(student.data, alpha=1.0 - momentum)

    def forward(self, views: list[torch.Tensor], teacher_temperature: float):
        student_logits = [
            self.student_head(self.student_encoder(view)) / self.student_temperature
            for view in views
        ]
        with torch.no_grad():
            teacher_logits = [
                self.teacher_head(self.teacher_encoder(view)) for view in views[:2]
            ]
            teacher_probabilities = [
                F.softmax((logits - self.center) / teacher_temperature, dim=-1)
                for logits in teacher_logits
            ]

        losses = []
        for teacher_index, targets in enumerate(teacher_probabilities):
            for student_index, logits in enumerate(student_logits):
                if student_index == teacher_index:
                    continue
                losses.append(
                    torch.sum(-targets * F.log_softmax(logits, dim=-1), dim=-1).mean()
                )
        loss = torch.stack(losses).mean()
        with torch.no_grad():
            batch_center = torch.cat(teacher_logits).mean(dim=0, keepdim=True)
            self.center.mul_(self.center_momentum).add_(
                batch_center, alpha=1.0 - self.center_momentum
            )
            entropy = torch.stack(
                [-(prob * prob.clamp_min(1e-8).log()).sum(dim=1).mean() for prob in teacher_probabilities]
            ).mean()
        return loss, entropy.detach()

