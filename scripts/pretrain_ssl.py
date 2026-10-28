#!/usr/bin/env python
"""Fixed-budget MoCo or DINO pretraining with representation diagnostics."""

from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import ManifestImageDataset, SSLManifestDataset  # noqa: E402
from src.data.manifest_dataset import build_transforms  # noqa: E402
from src.models import build_encoder, count_parameters  # noqa: E402
from src.ssl_methods import DINO, MoCo  # noqa: E402
from src.training import extract_features, feature_statistics, knn_classification  # noqa: E402


def resolve(value):
    path = Path(value); return path if path.is_absolute() else ROOT / path


def write_csv(path, rows):
    if not rows: return
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle: json.dump(value, handle, indent=2); handle.write("\n")
    temp.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): value.update(chunk)
    return value.hexdigest()


def schedule(epoch, epochs, warmup, minimum_ratio):
    if warmup and epoch < warmup: return (epoch + 1) / warmup
    progress = min(1.0, (epoch - warmup) / max(1, epochs - warmup - 1))
    return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda")); args = parser.parse_args()
    with resolve(args.config).open(encoding="utf-8") as handle: config = yaml.safe_load(handle)
    seed = int(config["seed"]); os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda: torch.backends.cudnn.benchmark = True; torch.set_float32_matmul_precision("high")

    run_name = config["run_name"]; result_dir = resolve(config["output_root"]) / run_name; checkpoint_dir = resolve(config["checkpoint_root"]) / run_name
    result_dir.mkdir(parents=True, exist_ok=True); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "config.yaml").open("w", encoding="utf-8") as handle: yaml.safe_dump(config, handle, sort_keys=False)
    dataset = SSLManifestDataset(
        resolve(config["data"]["train_root"]), resolve(config["data"]["train_manifest"]),
        method=config["method"], image_size=int(config["image_size"]),
        mean=config["data"]["normalization_mean"], std=config["data"]["normalization_std"],
        local_crops=int(config.get("local_crops", 2)), deduplicate_hashes=True,
    )
    workers = int(config["training"]["num_workers"])
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=True, num_workers=workers, pin_memory=use_cuda, persistent_workers=workers > 0, drop_last=True, generator=torch.Generator().manual_seed(seed))
    encoder, feature_dim = build_encoder(config["backbone"], image_size=int(config["image_size"]))
    if config["method"] == "moco":
        model = MoCo(encoder, feature_dim, projection_dim=int(config["projection_dim"]), projection_hidden_dim=int(config["projection_hidden_dim"]), queue_size=int(config["queue_size"]), momentum=float(config["momentum"]), temperature=float(config["temperature"]))
    else:
        model = DINO(encoder, feature_dim, hidden_dim=int(config["projection_hidden_dim"]), bottleneck_dim=int(config["bottleneck_dim"]), output_dim=int(config["output_dim"]), student_temperature=float(config["student_temperature"]), center_momentum=float(config["center_momentum"]))
    model.to(device); trainable = [p for p in model.parameters() if p.requires_grad]
    base_lr = float(config["training"]["learning_rate"]); epochs = int(config["training"]["epochs"]); warmup = int(config["training"]["warmup_epochs"])
    optimizer = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: schedule(e, epochs, warmup, float(config["training"]["minimum_learning_rate"]) / base_lr))
    amp = bool(config["training"]["mixed_precision"]) and use_cuda; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []; start_epoch = 0; last_path = checkpoint_dir / "last.pt"
    if last_path.exists() and not (result_dir / "summary.json").exists():
        state = torch.load(last_path, map_location=device, weights_only=False); model.load_state_dict(state["model_state"]); optimizer.load_state_dict(state["optimizer_state"]); scheduler.load_state_dict(state["scheduler_state"]); scaler.load_state_dict(state["scaler_state"]); history = state["history"]; start_epoch = state["epoch"]
        print(f"Resuming at epoch {start_epoch + 1}")
    total_parameters, trainable_parameters = count_parameters(model); started = time.perf_counter()
    if use_cuda: torch.cuda.reset_peak_memory_stats(device)
    print(f"{run_name}: {len(dataset)} hash-unique images; fixed {epochs}-epoch budget")
    for epoch in range(start_epoch, epochs):
        model.train(); total_loss = stat_a = stat_b = 0.0; batches = 0; epoch_started = time.perf_counter()
        for views, _ in tqdm(loader, desc=f"SSL {epoch + 1}/{epochs}", leave=False):
            views = [view.to(device, non_blocking=True) for view in views]; optimizer.zero_grad(set_to_none=True)
            teacher_momentum = teacher_temperature = 0.0
            if config["method"] == "dino":
                progress = (epoch + batches / max(1, len(loader))) / epochs
                teacher_momentum = 1 - (1 - float(config["teacher_momentum"])) * (math.cos(math.pi * progress) + 1) / 2
                temp_warmup = int(config["teacher_temperature_warmup_epochs"])
                if epoch < temp_warmup:
                    teacher_temperature = float(config["teacher_temperature_start"]) + (float(config["teacher_temperature"]) - float(config["teacher_temperature_start"])) * epoch / max(1, temp_warmup - 1)
                else: teacher_temperature = float(config["teacher_temperature"])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                if config["method"] == "moco": loss, first, second = model(views[0], views[1]); first, second = first.item(), second.item()
                else: loss, entropy = model(views, teacher_temperature); first, second = entropy.item(), teacher_momentum
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            if config["method"] == "dino" and epoch < int(config["freeze_last_layer_epochs"]):
                for parameter in model.student_head.last.parameters():
                    parameter.grad = None
            torch.nn.utils.clip_grad_norm_(trainable, float(config["training"]["gradient_clip_norm"])); scaler.step(optimizer); scaler.update()
            if config["method"] == "dino": model.update_teacher(teacher_momentum)
            total_loss += loss.item(); stat_a += first; stat_b += second; batches += 1
        history.append({"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"], "ssl_loss": total_loss / batches, "statistic_a": stat_a / batches, "statistic_b": stat_b / batches, "epoch_seconds": time.perf_counter() - epoch_started})
        write_csv(result_dir / "history.csv", history); scheduler.step()
        torch.save({"epoch": epoch + 1, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(), "history": history}, last_path)

    trained_encoder = model.query_encoder if config["method"] == "moco" else model.student_encoder
    encoder_path = checkpoint_dir / "encoder.pt"
    torch.save({"encoder_state": trained_encoder.state_dict(), "backbone": config["backbone"], "feature_dim": feature_dim, "image_size": int(config["image_size"]), "method": config["method"], "config": config}, encoder_path)
    _, clean_transform = build_transforms(int(config["image_size"]), config["data"]["normalization_mean"], config["data"]["normalization_std"])
    common = {"batch_size": 128, "shuffle": False, "num_workers": workers, "pin_memory": use_cuda}
    train_clean = DataLoader(ManifestImageDataset(resolve(config["data"]["train_root"]), resolve(config["data"]["train_manifest"]), clean_transform), **common)
    val_clean = DataLoader(ManifestImageDataset(resolve(config["data"]["train_root"]), resolve(config["data"]["validation_manifest"]), clean_transform), **common)
    train_features, train_labels = extract_features(trained_encoder, train_clean, device, "Train features"); val_features, val_labels = extract_features(trained_encoder, val_clean, device, "Validation features")
    diagnostics = feature_statistics(train_features); diagnostics.update(knn_classification(train_features, train_labels, val_features, val_labels, num_classes=16)); write_json(result_dir / "diagnostics.json", diagnostics)
    summary = {"run_name": run_name, "method": config["method"], "backbone": config["backbone"], "external_pretraining": False, "epochs_completed": epochs, "pretraining_images_after_hash_deduplication": len(dataset), "total_parameters": total_parameters, "trainable_parameters": trainable_parameters, "total_seconds": time.perf_counter() - started, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if use_cuda else 0.0, "encoder_checkpoint": str(encoder_path.relative_to(ROOT)), "encoder_sha256": digest(encoder_path), "diagnostics": diagnostics, "test_evaluated": False}
    write_json(result_dir / "summary.json", summary); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
