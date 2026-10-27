#!/usr/bin/env python
"""Train a supervised control without loading or evaluating either test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import torchvision
import yaml
from sklearn.metrics import confusion_matrix
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders  # noqa: E402
from src.models import build_model, count_parameters  # noqa: E402
from src.training import evaluate, train_one_epoch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    return parser.parse_args()


def absolute(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int,
    minimum_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def schedule(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        denominator = max(1, epochs - warmup_epochs - 1)
        progress = min(1.0, max(0.0, (epoch - warmup_epochs) / denominator))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "sklearn": sklearn.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def main() -> None:
    args = parse_args()
    config_path = absolute(args.config)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["seed"])
    seed_everything(seed)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    run_name = str(config["run_name"])
    result_dir = absolute(config["output_root"]) / run_name
    checkpoint_dir = absolute(config["checkpoint_root"]) / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / "best.pt"
    last_checkpoint = checkpoint_dir / "last.pt"

    with (result_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    train_loader, validation_loader, class_names = build_dataloaders(
        train_root=absolute(config["data"]["train_root"]),
        train_manifest=absolute(config["data"]["train_manifest"]),
        validation_manifest=absolute(config["data"]["validation_manifest"]),
        image_size=int(config["model"]["image_size"]),
        mean=list(config["data"]["normalization_mean"]),
        std=list(config["data"]["normalization_std"]),
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        seed=seed,
        device_type=device.type,
    )
    if len(class_names) != int(config["model"]["num_classes"]):
        raise RuntimeError("Configured class count does not match the manifests")

    model = build_model(
        str(config["model"]["name"]),
        num_classes=int(config["model"]["num_classes"]),
        image_size=int(config["model"]["image_size"]),
    ).to(device)
    total_parameters, trainable_parameters = count_parameters(model)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["training"]["label_smoothing"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    minimum_lr_ratio = (
        float(config["training"]["minimum_learning_rate"])
        / float(config["training"]["learning_rate"])
    )
    scheduler = cosine_with_warmup(
        optimizer,
        epochs=int(config["training"]["epochs"]),
        warmup_epochs=int(config["training"]["warmup_epochs"]),
        minimum_lr_ratio=minimum_lr_ratio,
    )
    amp_enabled = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print("=" * 72)
    print(f"Run:             {run_name}")
    print(f"Model:           {config['model']['name']} (random initialization)")
    print(f"Device:          {device}")
    print(f"Train/val:       {len(train_loader.dataset)}/{len(validation_loader.dataset)}")
    print(f"Parameters:      {total_parameters:,} total; {trainable_parameters:,} trainable")
    print("Test evaluation: disabled during model selection")
    print("=" * 72)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_macro_f1 = -1.0
    best_epoch = -1
    stale_epochs = 0
    epochs = int(config["training"]["epochs"])

    for epoch in range(epochs):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_output = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            epoch=epoch,
            epochs=epochs,
        )
        validation_output = evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            description=f"Validation {epoch + 1}/{epochs}",
        )
        row = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "train_loss": train_output.loss,
            "train_accuracy": train_output.accuracy,
            "train_macro_f1": train_output.macro_f1,
            "validation_loss": validation_output.loss,
            "validation_accuracy": validation_output.accuracy,
            "validation_macro_f1": validation_output.macro_f1,
            "train_seconds": train_output.duration_seconds,
            "validation_seconds": validation_output.duration_seconds,
            "train_images_per_second": train_output.images_per_second,
            "validation_images_per_second": validation_output.images_per_second,
        }
        history.append(row)
        write_csv(result_dir / "history.csv", history)

        improved = (
            validation_output.accuracy > best_accuracy
            or (
                validation_output.accuracy == best_accuracy
                and validation_output.macro_f1 > best_macro_f1
            )
        )
        checkpoint = {
            "model_state": model.state_dict(),
            "model_name": config["model"]["name"],
            "num_classes": config["model"]["num_classes"],
            "image_size": config["model"]["image_size"],
            "class_names": class_names,
            "normalization_mean": config["data"]["normalization_mean"],
            "normalization_std": config["data"]["normalization_std"],
            "epoch": epoch + 1,
            "validation_accuracy": validation_output.accuracy,
            "validation_macro_f1": validation_output.macro_f1,
            "config": config,
        }
        torch.save(checkpoint, last_checkpoint)
        if improved:
            best_accuracy = validation_output.accuracy
            best_macro_f1 = validation_output.macro_f1
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save(checkpoint, best_checkpoint)
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch + 1:03d}: "
            f"train acc={train_output.accuracy:.4f}, "
            f"val acc={validation_output.accuracy:.4f}, "
            f"val F1={validation_output.macro_f1:.4f}, "
            f"best={best_accuracy:.4f} (epoch {best_epoch}), "
            f"time={train_output.duration_seconds + validation_output.duration_seconds:.1f}s"
        )
        scheduler.step()
        if stale_epochs >= int(config["training"]["early_stopping_patience"]):
            print(f"Early stopping after {stale_epochs} epochs without improvement")
            break

    total_seconds = time.perf_counter() - started
    trusted_checkpoint = torch.load(
        best_checkpoint,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(trusted_checkpoint["model_state"])
    final_validation = evaluate(
        model=model,
        loader=validation_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
        description="Best-checkpoint validation",
    )
    matrix = confusion_matrix(
        final_validation.labels,
        final_validation.predictions,
        labels=list(range(len(class_names))),
    )
    with (result_dir / "validation_confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *class_names])
        for class_name, values in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *values])

    peak_memory_mib = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    summary = {
        "run_name": run_name,
        "method": "supervised",
        "external_pretraining": False,
        "model": config["model"]["name"],
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "best_validation_macro_f1": best_macro_f1,
        "final_best_checkpoint_validation_accuracy": final_validation.accuracy,
        "final_best_checkpoint_validation_macro_f1": final_validation.macro_f1,
        "train_images": len(train_loader.dataset),
        "validation_images": len(validation_loader.dataset),
        "test_evaluated": False,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "total_seconds": total_seconds,
        "peak_gpu_memory_mib": peak_memory_mib,
        "checkpoint_path": str(best_checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": file_sha256(best_checkpoint),
        "environment": environment_metadata(device),
    }
    write_json(result_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

