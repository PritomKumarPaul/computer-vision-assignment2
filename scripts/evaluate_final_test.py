#!/usr/bin/env python
"""One-time evaluation of the frozen final model on the labeled test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifest_dataset import build_transforms  # noqa: E402
from src.models import EncoderClassifier, build_encoder  # noqa: E402
from src.training import evaluate  # noqa: E402


FINAL_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "checkpoints"
    / "checkpoint2"
    / "classification"
    / "dino_resnet18_full_finetune_seed0"
    / "best.pt"
)
FINAL_CHECKPOINT_SHA256 = (
    "93bbf0877348a81b167ee5fe0bedb475e3ed9aeff0d0754c644ec6a5b8976d95"
)
TEST_ROOT = ROOT / "data" / "test"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-final-selection",
        action="store_true",
        help="Confirm that model selection is frozen before test evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "submission" / "final_test",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    if not args.confirm_final_selection:
        raise RuntimeError(
            "Refusing test access without --confirm-final-selection. "
            "The final model must be frozen before evaluating the test set."
        )
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        raise RuntimeError(
            f"Final test output already exists at {metrics_path}; refusing repeated evaluation."
        )
    if not FINAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Final checkpoint is missing: {FINAL_CHECKPOINT}. "
            "Restore it from the checkpoint3 GitHub Release."
        )
    observed_digest = sha256(FINAL_CHECKPOINT)
    if observed_digest != FINAL_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Final checkpoint checksum does not match the model selected before test access"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    checkpoint = torch.load(FINAL_CHECKPOINT, map_location="cpu", weights_only=False)
    encoder, feature_dim = build_encoder(
        checkpoint["model_name"], image_size=int(checkpoint["image_size"])
    )
    model = EncoderClassifier(
        encoder,
        feature_dim,
        int(checkpoint["num_classes"]),
        freeze_encoder=False,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)

    _, clean_transform = build_transforms(
        int(checkpoint["image_size"]),
        list(checkpoint["normalization_mean"]),
        list(checkpoint["normalization_std"]),
    )
    dataset = ImageFolder(TEST_ROOT, transform=clean_transform)
    if dataset.classes != checkpoint["class_names"]:
        raise RuntimeError(
            f"Test class mapping {dataset.classes} does not match checkpoint "
            f"mapping {checkpoint['class_names']}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    output = evaluate(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        amp_enabled=device.type == "cuda",
        description="FINAL labeled-test evaluation",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(
        output.labels,
        output.predictions,
        labels=list(range(len(dataset.classes))),
    )
    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *dataset.classes])
        for class_name, values in zip(dataset.classes, matrix.tolist()):
            writer.writerow([class_name, *values])

    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "actual", "predicted", "correct"])
        for (path, actual), predicted in zip(dataset.samples, output.predictions):
            writer.writerow(
                [
                    str(Path(path).relative_to(TEST_ROOT)),
                    dataset.classes[actual],
                    dataset.classes[predicted],
                    actual == predicted,
                ]
            )

    per_class_recall = {
        class_name: float(matrix[index, index] / max(1, matrix[index].sum()))
        for index, class_name in enumerate(dataset.classes)
    }
    metrics = {
        "evaluation": "one-time final labeled-test evaluation",
        "model_selection_frozen_before_test": True,
        "run_name": checkpoint["config"]["run_name"],
        "method": "DINO pretraining followed by full ResNet-18 fine-tuning",
        "checkpoint": str(FINAL_CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": observed_digest,
        "selection_validation_accuracy": checkpoint["validation_accuracy"],
        "selection_validation_macro_f1": checkpoint["validation_macro_f1"],
        "test_images": len(dataset),
        "test_accuracy": output.accuracy,
        "test_macro_f1": output.macro_f1,
        "test_cross_entropy": output.loss,
        "per_class_recall": per_class_recall,
        "duration_seconds": output.duration_seconds,
        "device": str(device),
        "test_evaluated": True,
        "test2_evaluated": False,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
