"""Representation diagnostics shared by all SSL methods and backbones."""

from __future__ import annotations

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@torch.inference_mode()
def extract_features(encoder: nn.Module, loader: DataLoader, device: torch.device, description: str):
    encoder.eval(); features, labels = [], []
    for images, targets in tqdm(loader, desc=description, leave=False):
        features.append(encoder(images.to(device, non_blocking=True)).float().cpu())
        labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)


def feature_statistics(features: torch.Tensor) -> dict[str, float]:
    centered = features - features.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False)
    spectrum = torch.linalg.svdvals(centered).square()
    spectrum = spectrum / spectrum.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(spectrum * spectrum.clamp_min(1e-12).log()).sum()).item()
    normalized = F.normalize(features, dim=1)
    gram = normalized @ normalized.T
    count = gram.shape[0]
    mean_cosine = ((gram.sum() - count) / max(1, count * (count - 1))).item()
    return {
        "mean_dimension_std": std.mean().item(),
        "minimum_dimension_std": std.min().item(),
        "effective_rank": effective_rank,
        "mean_pairwise_cosine_similarity": mean_cosine,
    }


def knn_classification(train_features, train_labels, validation_features, validation_labels, *, num_classes: int, neighbors: int = 5, temperature: float = 0.1):
    train_features = F.normalize(train_features, dim=1)
    validation_features = F.normalize(validation_features, dim=1)
    values, indices = (validation_features @ train_features.T).topk(min(neighbors, len(train_features)), dim=1)
    scores = torch.zeros(len(validation_features), num_classes)
    scores.scatter_add_(1, train_labels[indices], torch.exp(values / temperature))
    predictions = scores.argmax(dim=1).numpy(); labels = validation_labels.numpy()
    return {
        "knn_accuracy": float(accuracy_score(labels, predictions)),
        "knn_macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "knn_neighbors": neighbors,
    }

