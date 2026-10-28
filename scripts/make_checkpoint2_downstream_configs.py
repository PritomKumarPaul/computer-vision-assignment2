#!/usr/bin/env python
"""Generate the eight resolved probe/fine-tune configurations."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "checkpoint2_configs"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for method in ("moco", "dino"):
        for backbone in ("resnet18", "vit_tiny"):
            pretraining_run = f"{method}_{backbone}_seed0"
            for protocol in ("linear_probe", "full_finetune"):
                run_name = f"{method}_{backbone}_{protocol}_seed0"
                learning_rate = 0.001 if protocol == "linear_probe" else (
                    0.0001 if backbone == "resnet18" else 0.00005
                )
                config = {
                    "run_name": run_name,
                    "seed": 0,
                    "output_root": "results/checkpoint2/classification",
                    "checkpoint_root": "artifacts/checkpoints/checkpoint2/classification",
                    "data": {
                        "train_root": "data/train",
                        "train_manifest": "splits/train.csv",
                        "validation_manifest": "splits/val.csv",
                        "normalization_mean": [0.5, 0.5, 0.5],
                        "normalization_std": [0.25, 0.25, 0.25],
                    },
                    "model": {
                        "name": backbone,
                        "num_classes": 16,
                        "image_size": 224,
                        "initialization": "ssl",
                        "ssl_method": method,
                        "ssl_checkpoint": f"artifacts/checkpoints/checkpoint2/pretraining/{pretraining_run}/encoder.pt",
                    },
                    "training": {
                        "protocol": protocol,
                        "batch_size": 64,
                        "num_workers": 8,
                        "epochs": 100,
                        "early_stopping_patience": 15 if protocol == "linear_probe" else 20,
                        "learning_rate": learning_rate,
                        "head_learning_rate": 0.001,
                        "minimum_learning_rate": 0.000001,
                        "warmup_epochs": 5,
                        "weight_decay": 0.0001 if backbone == "resnet18" else 0.05,
                        "label_smoothing": 0.0 if protocol == "linear_probe" else 0.1,
                        "gradient_clip_norm": 1.0,
                        "mixed_precision": True,
                    },
                }
                with (OUTPUT / f"{run_name}.yaml").open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(config, handle, sort_keys=False)
    print(f"Wrote eight downstream configurations to {OUTPUT}")


if __name__ == "__main__":
    main()

