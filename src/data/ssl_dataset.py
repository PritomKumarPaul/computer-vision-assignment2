"""Multi-view dataset used by MoCo and DINO pretraining."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _view_transform(
    image_size: int,
    scale: tuple[float, float],
    mean: list[float],
    std: list[float],
    blur_probability: float,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=scale,
                ratio=(0.80, 1.20),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.25, contrast=0.25)],
                p=0.6,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
                p=blur_probability,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class SSLManifestDataset(Dataset):
    """Return multiple label-free views while retaining labels for diagnostics."""

    def __init__(
        self,
        image_root: Path,
        manifest_path: Path,
        *,
        method: str,
        image_size: int,
        mean: list[float],
        std: list[float],
        local_crops: int = 2,
        deduplicate_hashes: bool = True,
    ) -> None:
        self.image_root = Path(image_root)
        with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if deduplicate_hashes:
            unique = {}
            for row in rows:
                unique.setdefault(row["sha256"], row)
            rows = list(unique.values())
        self.rows = sorted(rows, key=lambda row: row["relative_path"])
        self.method = method
        self.local_crops = local_crops if method == "dino" else 0
        self.global_transform = _view_transform(
            image_size, (0.50, 1.0), mean, std, 0.35
        )
        self.local_transform = _view_transform(
            image_size, (0.25, 0.55), mean, std, 0.15
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[list[torch.Tensor], int]:
        row = self.rows[index]
        with Image.open(self.image_root / row["relative_path"]) as image:
            image = image.convert("RGB")
            views = [self.global_transform(image), self.global_transform(image)]
            views.extend(self.local_transform(image) for _ in range(self.local_crops))
        return views, int(row["label"])

