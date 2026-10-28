"""A compact MoCo-v2-style objective for either supported encoder."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


def projection_head(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


class MoCo(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        *,
        projection_dim: int,
        projection_hidden_dim: int,
        queue_size: int,
        momentum: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.query_encoder = encoder
        self.query_projector = projection_head(
            feature_dim, projection_hidden_dim, projection_dim
        )
        self.key_encoder = copy.deepcopy(encoder)
        self.key_projector = copy.deepcopy(self.query_projector)
        for parameter in list(self.key_encoder.parameters()) + list(
            self.key_projector.parameters()
        ):
            parameter.requires_grad = False
        self.momentum = momentum
        self.temperature = temperature
        self.register_buffer("queue", F.normalize(torch.randn(projection_dim, queue_size), dim=0))
        self.register_buffer("queue_pointer", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_key_network(self) -> None:
        for query, key in zip(
            list(self.query_encoder.parameters()) + list(self.query_projector.parameters()),
            list(self.key_encoder.parameters()) + list(self.key_projector.parameters()),
        ):
            key.data.mul_(self.momentum).add_(query.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        batch = keys.shape[0]
        size = self.queue.shape[1]
        pointer = int(self.queue_pointer.item())
        if batch >= size:
            self.queue.copy_(keys[-size:].T)
            pointer = 0
        else:
            first = min(batch, size - pointer)
            self.queue[:, pointer:pointer + first] = keys[:first].T
            remainder = batch - first
            if remainder:
                self.queue[:, :remainder] = keys[first:].T
            pointer = (pointer + batch) % size
        self.queue_pointer[0] = pointer

    def forward(self, query_view: torch.Tensor, key_view: torch.Tensor):
        query = F.normalize(
            self.query_projector(self.query_encoder(query_view)), dim=1
        )
        with torch.no_grad():
            self.update_key_network()
            key = F.normalize(self.key_projector(self.key_encoder(key_view)), dim=1)
        positive = torch.einsum("nd,nd->n", query, key).unsqueeze(1)
        negative = torch.einsum("nd,dk->nk", query, self.queue.detach())
        logits = torch.cat([positive, negative], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        self.enqueue(key)
        return loss, positive.mean().detach(), negative.mean().detach()

