"""Shared epoch loops for supervised controls and later fine-tuning."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class EpochOutput:
    loss: float
    accuracy: float
    macro_f1: float
    duration_seconds: float
    images_per_second: float
    labels: list[int]
    predictions: list[int]


def _metrics(
    *,
    total_loss: float,
    sample_count: int,
    labels: list[int],
    predictions: list[int],
    duration: float,
) -> EpochOutput:
    return EpochOutput(
        loss=total_loss / sample_count,
        accuracy=float(accuracy_score(labels, predictions)),
        macro_f1=float(f1_score(labels, predictions, average="macro", zero_division=0)),
        duration_seconds=duration,
        images_per_second=sample_count / max(duration, 1e-9),
        labels=labels,
        predictions=predictions,
    )


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    gradient_clip_norm: float,
    epoch: int,
    epochs: int,
) -> EpochOutput:
    model.train()
    total_loss = 0.0
    sample_count = 0
    labels: list[int] = []
    predictions: list[int] = []
    started = time.perf_counter()

    progress = tqdm(loader, desc=f"Train {epoch + 1}/{epochs}", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.shape[0]
        total_loss += loss.detach().item() * batch_size
        sample_count += batch_size
        labels.extend(targets.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
        progress.set_postfix(loss=f"{total_loss / sample_count:.4f}")

    duration = time.perf_counter() - started
    return _metrics(
        total_loss=total_loss,
        sample_count=sample_count,
        labels=labels,
        predictions=predictions,
        duration=duration,
    )


@torch.inference_mode()
def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    description: str,
) -> EpochOutput:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    labels: list[int] = []
    predictions: list[int] = []
    started = time.perf_counter()

    progress = tqdm(loader, desc=description, leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size
        labels.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())

    duration = time.perf_counter() - started
    return _metrics(
        total_loss=total_loss,
        sample_count=sample_count,
        labels=labels,
        predictions=predictions,
        duration=duration,
    )

