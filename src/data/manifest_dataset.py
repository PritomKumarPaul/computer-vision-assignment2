"""Manifest-backed datasets for leakage-controlled supervised experiments."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ManifestImageDataset(Dataset):
    """Load labeled images using a fixed CSV manifest."""

    def __init__(
        self,
        image_root: Path,
        manifest_path: Path,
        transform: Any,
    ) -> None:
        self.image_root = Path(image_root)
        self.manifest_path = Path(manifest_path)
        self.transform = transform

        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise RuntimeError(f"Manifest contains no images: {self.manifest_path}")

        self.class_names = [
            name
            for _, name in sorted(
                {(int(row["label"]), row["class_name"]) for row in self.rows}
            )
        ]
        expected_labels = list(range(len(self.class_names)))
        observed_labels = sorted({int(row["label"]) for row in self.rows})
        if observed_labels != expected_labels:
            raise ValueError(
                f"Labels must be contiguous from zero; observed {observed_labels}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = self.image_root / row["relative_path"]
        with Image.open(path) as image:
            # Every input has the same tensor shape. Grayscale source images are
            # replicated across RGB channels; no external color information is
            # introduced.
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["label"])


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_transforms(
    image_size: int,
    mean: list[float],
    std: list[float],
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return matched scene-preserving train and evaluation transforms."""

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.65, 1.0),
                ratio=(0.85, 1.15),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.20, contrast=0.20)],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                round(image_size * 256 / 224),
                antialias=True,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(
    *,
    train_root: Path,
    train_manifest: Path,
    validation_manifest: Path,
    image_size: int,
    mean: list[float],
    std: list[float],
    batch_size: int,
    num_workers: int,
    seed: int,
    device_type: str,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Build train/validation loaders without exposing either test split."""

    train_transform, eval_transform = build_transforms(image_size, mean, std)
    train_dataset = ManifestImageDataset(
        train_root, train_manifest, train_transform
    )
    validation_dataset = ManifestImageDataset(
        train_root, validation_manifest, eval_transform
    )
    if train_dataset.class_names != validation_dataset.class_names:
        raise RuntimeError("Train and validation class mappings differ")

    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device_type == "cuda",
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _worker_seed,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, validation_loader, train_dataset.class_names

