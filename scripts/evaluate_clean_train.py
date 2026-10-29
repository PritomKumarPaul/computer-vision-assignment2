#!/usr/bin/env python
"""Evaluate a best checkpoint on clean training views; test access is forbidden."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifest_dataset import (  # noqa: E402
    ManifestImageDataset,
    build_transforms,
)
from src.models import (  # noqa: E402
    EncoderClassifier,
    MAEAuxiliaryModel,
    build_encoder,
    build_model,
)
from src.training import evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    checkpoint_path = resolve(args.checkpoint)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model_config = checkpoint.get("config", {}).get("model", {})
    if checkpoint.get("task") == "joint_mae":
        model = MAEAuxiliaryModel(
            checkpoint["model_name"],
            num_classes=int(checkpoint["num_classes"]),
            image_size=int(checkpoint["image_size"]),
            patch_size=int(checkpoint["patch_size"]),
        )
    elif model_config.get("initialization", "random") == "ssl":
        encoder, feature_dim = build_encoder(
            checkpoint["model_name"], image_size=int(checkpoint["image_size"])
        )
        model = EncoderClassifier(
            encoder,
            feature_dim,
            int(checkpoint["num_classes"]),
            freeze_encoder=checkpoint["config"]["training"]["protocol"] == "linear_probe",
        )
    else:
        model = build_model(
            checkpoint["model_name"],
            num_classes=int(checkpoint["num_classes"]),
            image_size=int(checkpoint["image_size"]),
        )
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state"])

    _, clean_transform = build_transforms(
        int(checkpoint["image_size"]),
        list(checkpoint["normalization_mean"]),
        list(checkpoint["normalization_std"]),
    )
    # This script intentionally hard-codes the labeled training partition and
    # exposes no CLI option for test or test2 paths.
    dataset = ManifestImageDataset(
        ROOT / "data" / "train",
        ROOT / "splits" / "train.csv",
        clean_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    label_smoothing = float(
        checkpoint.get("config", {})
        .get("training", {})
        .get("label_smoothing", 0.0)
    )
    output = evaluate(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(label_smoothing=label_smoothing),
        device=device,
        amp_enabled=device.type == "cuda",
        description="Clean training diagnostic",
    )
    metrics = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "split": "fixed training manifest with deterministic evaluation transform",
        "images": len(dataset),
        "loss": output.loss,
        "accuracy": output.accuracy,
        "macro_f1": output.macro_f1,
        "duration_seconds": output.duration_seconds,
        "test_evaluated": False,
    }
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

