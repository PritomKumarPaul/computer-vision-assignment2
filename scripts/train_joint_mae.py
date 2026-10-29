#!/usr/bin/env python
"""Joint scene classification and MAE-style masked reconstruction training."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_supervised import (  # noqa: E402
    absolute,
    cosine_with_warmup,
    environment_metadata,
    file_sha256,
    seed_everything,
    select_device,
    write_csv,
    write_json,
)
from src.data import build_dataloaders  # noqa: E402
from src.models import MAEAuxiliaryModel, count_parameters  # noqa: E402
from src.training import evaluate, mask_images, masked_patch_mse  # noqa: E402


def train_joint_epoch(
    *,
    model: MAEAuxiliaryModel,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    patch_size: int,
    mask_ratio: float,
    classification_weight: float,
    reconstruction_weight: float,
    gradient_clip_norm: float,
    epoch: int,
    epochs: int,
) -> dict[str, Any]:
    model.train()
    totals = {"total": 0.0, "classification": 0.0, "reconstruction": 0.0}
    sample_count = 0
    labels: list[int] = []
    predictions: list[int] = []
    started = time.perf_counter()

    progress = tqdm(loader, desc=f"Joint MAE {epoch + 1}/{epochs}", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masked_images, patch_targets, patch_mask = mask_images(
            images,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            patch_predictions = model.reconstruct(masked_images)
            classification_loss = criterion(logits, targets)
            reconstruction_loss = masked_patch_mse(
                patch_predictions, patch_targets, patch_mask
            )
            total_loss = (
                classification_weight * classification_loss
                + reconstruction_weight * reconstruction_loss
            )

        scaler.scale(total_loss).backward()
        if gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.shape[0]
        totals["total"] += total_loss.detach().item() * batch_size
        totals["classification"] += classification_loss.detach().item() * batch_size
        totals["reconstruction"] += reconstruction_loss.detach().item() * batch_size
        sample_count += batch_size
        labels.extend(targets.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
        progress.set_postfix(
            total=f"{totals['total'] / sample_count:.4f}",
            mae=f"{totals['reconstruction'] / sample_count:.4f}",
        )

    duration = time.perf_counter() - started
    return {
        "total_loss": totals["total"] / sample_count,
        "classification_loss": totals["classification"] / sample_count,
        "reconstruction_loss": totals["reconstruction"] / sample_count,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "duration_seconds": duration,
        "images_per_second": sample_count / max(duration, 1e-9),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda")
    )
    args = parser.parse_args()

    with absolute(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed = int(config["seed"])
    seed_everything(seed)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    objective = config["objective"]
    classification_weight = float(objective["classification_weight"])
    reconstruction_weight = float(objective["reconstruction_weight"])
    if abs(classification_weight + reconstruction_weight - 1.0) > 1e-8:
        raise ValueError("Classification and reconstruction weights must sum to one")

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

    model = MAEAuxiliaryModel(
        str(config["model"]["name"]),
        num_classes=int(config["model"]["num_classes"]),
        image_size=int(config["model"]["image_size"]),
        patch_size=int(objective["patch_size"]),
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
    scheduler = cosine_with_warmup(
        optimizer,
        epochs=int(config["training"]["epochs"]),
        warmup_epochs=int(config["training"]["warmup_epochs"]),
        minimum_lr_ratio=float(config["training"]["minimum_learning_rate"])
        / float(config["training"]["learning_rate"]),
    )
    amp_enabled = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print("=" * 72)
    print(f"Run:             {run_name}")
    print(f"Backbone:        {config['model']['name']} (random initialization)")
    print(
        f"Objective:       {classification_weight:.2f} classification + "
        f"{reconstruction_weight:.2f} masked reconstruction"
    )
    print(
        f"Masking:         {float(objective['mask_ratio']):.0%} of "
        f"{int(objective['patch_size'])}x{int(objective['patch_size'])} patches"
    )
    print(f"Device:          {device}")
    print(f"Train/val:       {len(train_loader.dataset)}/{len(validation_loader.dataset)}")
    print(f"Parameters:      {total_parameters:,} total; {trainable_parameters:,} trainable")
    print("Model selection: clean validation classification only")
    print("Test evaluation: disabled")
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
        train_output = train_joint_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            patch_size=int(objective["patch_size"]),
            mask_ratio=float(objective["mask_ratio"]),
            classification_weight=classification_weight,
            reconstruction_weight=reconstruction_weight,
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
            "train_total_loss": train_output["total_loss"],
            "train_classification_loss": train_output["classification_loss"],
            "train_mae_loss": train_output["reconstruction_loss"],
            "weighted_classification_loss": classification_weight
            * train_output["classification_loss"],
            "weighted_mae_loss": reconstruction_weight
            * train_output["reconstruction_loss"],
            "train_accuracy": train_output["accuracy"],
            "train_macro_f1": train_output["macro_f1"],
            "validation_classification_loss": validation_output.loss,
            "validation_accuracy": validation_output.accuracy,
            "validation_macro_f1": validation_output.macro_f1,
            "train_seconds": train_output["duration_seconds"],
            "validation_seconds": validation_output.duration_seconds,
            "train_images_per_second": train_output["images_per_second"],
            "validation_images_per_second": validation_output.images_per_second,
        }
        history.append(row)
        write_csv(result_dir / "history.csv", history)

        improved = validation_output.accuracy > best_accuracy or (
            validation_output.accuracy == best_accuracy
            and validation_output.macro_f1 > best_macro_f1
        )
        checkpoint = {
            "model_state": model.state_dict(),
            "model_name": config["model"]["name"],
            "task": "joint_mae",
            "num_classes": config["model"]["num_classes"],
            "image_size": config["model"]["image_size"],
            "patch_size": objective["patch_size"],
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
            f"Epoch {epoch + 1:03d}: total={train_output['total_loss']:.4f}, "
            f"CE={train_output['classification_loss']:.4f}, "
            f"MAE={train_output['reconstruction_loss']:.4f}, "
            f"train acc={train_output['accuracy']:.4f}, "
            f"val acc={validation_output.accuracy:.4f}, "
            f"best={best_accuracy:.4f} (epoch {best_epoch}), "
            f"time={train_output['duration_seconds'] + validation_output.duration_seconds:.1f}s"
        )
        scheduler.step()
        if stale_epochs >= int(config["training"]["early_stopping_patience"]):
            print(f"Early stopping after {stale_epochs} epochs without improvement")
            break

    total_seconds = time.perf_counter() - started
    trusted_checkpoint = torch.load(
        best_checkpoint, map_location=device, weights_only=False
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
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    summary = {
        "run_name": run_name,
        "method": "joint_mae_auxiliary",
        "external_pretraining": False,
        "model": config["model"]["name"],
        "seed": seed,
        "classification_weight": classification_weight,
        "reconstruction_weight": reconstruction_weight,
        "mask_ratio": float(objective["mask_ratio"]),
        "patch_size": int(objective["patch_size"]),
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
