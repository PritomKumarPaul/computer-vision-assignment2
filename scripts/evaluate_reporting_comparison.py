#!/usr/bin/env python
"""One-time test evaluation of preselected models required for the report table."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifest_dataset import build_transforms  # noqa: E402
from src.models import (  # noqa: E402
    EncoderClassifier,
    MAEAuxiliaryModel,
    build_encoder,
    build_model,
)
from src.training import evaluate  # noqa: E402


PRESELECTED_MODELS = (
    {
        "report_label": "Supervised baseline",
        "path": ROOT
        / "artifacts/checkpoints/checkpoint1/supervised_resnet18_seed0/best.pt",
        "sha256": "7a2263884c42e3c4caa6c3862c36bdf6413956b87cc3e3b323d0a4f6c41fefed",
    },
    {
        "report_label": "MoCo pretraining + full fine-tuning",
        "path": ROOT
        / "artifacts/checkpoints/checkpoint2/classification/moco_resnet18_full_finetune_seed0/best.pt",
        "sha256": "6457e8add8bebf3e04ca122e5efdb86d93647000de02ad725157604796872f49",
    },
    {
        "report_label": "Joint MAE + classification",
        "path": ROOT
        / "artifacts/checkpoints/checkpoint3/joint_mae_vit_tiny_seed0/best.pt",
        "sha256": "84950e74fd06dd6496ead126b52623afdefd04f3973008f9ed7b18ad06669f26",
    },
)
FINAL_METRICS = ROOT / "submission/final_test/metrics.json"
TEST_ROOT = ROOT / "data/test"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checkpoint_model(checkpoint: dict) -> nn.Module:
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    if checkpoint.get("task") == "joint_mae":
        return MAEAuxiliaryModel(
            checkpoint["model_name"],
            num_classes=int(checkpoint["num_classes"]),
            image_size=int(checkpoint["image_size"]),
            patch_size=int(checkpoint["patch_size"]),
        )
    if model_config.get("initialization") == "ssl":
        encoder, feature_dim = build_encoder(
            checkpoint["model_name"], image_size=int(checkpoint["image_size"])
        )
        return EncoderClassifier(
            encoder,
            feature_dim,
            int(checkpoint["num_classes"]),
            freeze_encoder=config["training"]["protocol"] == "linear_probe",
        )
    return build_model(
        checkpoint["model_name"],
        num_classes=int(checkpoint["num_classes"]),
        image_size=int(checkpoint["image_size"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-reporting-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "submission/reporting_test_comparison.json",
    )
    args = parser.parse_args()
    if not args.confirm_reporting_only:
        raise RuntimeError(
            "Refusing test access without --confirm-reporting-only. "
            "These checkpoints must be frozen and no further tuning is permitted."
        )
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    if output_path.exists():
        raise RuntimeError(
            f"Reporting comparison already exists at {output_path}; refusing repeated evaluation."
        )
    if not FINAL_METRICS.exists():
        raise FileNotFoundError("The locked final-model evaluation must be completed first")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    final_metrics = json.loads(FINAL_METRICS.read_text(encoding="utf-8"))
    rows = [
        {
            "report_label": "DINO pretraining + full fine-tuning (final)",
            "run_name": final_metrics["run_name"],
            "backbone": "resnet18",
            "test_accuracy": final_metrics["test_accuracy"],
            "test_macro_f1": final_metrics["test_macro_f1"],
            "checkpoint_sha256": final_metrics["checkpoint_sha256"],
            "evaluated_now": False,
        }
    ]

    reference_checkpoint = torch.load(
        PRESELECTED_MODELS[0]["path"], map_location="cpu", weights_only=False
    )
    _, clean_transform = build_transforms(
        int(reference_checkpoint["image_size"]),
        list(reference_checkpoint["normalization_mean"]),
        list(reference_checkpoint["normalization_std"]),
    )
    dataset = ImageFolder(TEST_ROOT, transform=clean_transform)
    if dataset.classes != reference_checkpoint["class_names"]:
        raise RuntimeError("Test class mapping does not match the frozen checkpoints")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    for selection in PRESELECTED_MODELS:
        path = selection["path"]
        if not path.exists():
            raise FileNotFoundError(f"Preselected reporting checkpoint is missing: {path}")
        observed_digest = sha256(path)
        if observed_digest != selection["sha256"]:
            raise RuntimeError(f"Checksum mismatch for frozen checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["class_names"] != dataset.classes:
            raise RuntimeError(f"Class mapping mismatch for {path}")
        model = build_checkpoint_model(checkpoint)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)
        output = evaluate(
            model=model,
            loader=loader,
            criterion=nn.CrossEntropyLoss(),
            device=device,
            amp_enabled=device.type == "cuda",
            description=f"REPORTING ONLY: {selection['report_label']}",
        )
        rows.append(
            {
                "report_label": selection["report_label"],
                "run_name": checkpoint["config"]["run_name"],
                "backbone": checkpoint["model_name"],
                "test_accuracy": output.accuracy,
                "test_macro_f1": output.macro_f1,
                "checkpoint_sha256": observed_digest,
                "evaluated_now": True,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "purpose": "final report comparison of checkpoints selected before test access",
        "no_further_model_selection_or_tuning": True,
        "test_images": len(dataset),
        "test2_evaluated": False,
        "models": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
