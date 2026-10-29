#!/usr/bin/env python
"""Create the checkpoint-3 joint-MAE comparison table."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    base = ROOT / "results" / "checkpoint3"
    for summary_path in sorted(base.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        clean_path = summary_path.parent / "clean_train_metrics.json"
        clean = (
            json.loads(clean_path.read_text(encoding="utf-8"))
            if clean_path.exists()
            else {}
        )
        rows.append(
            {
                "run_name": summary["run_name"],
                "backbone": summary["model"],
                "classification_weight": summary["classification_weight"],
                "reconstruction_weight": summary["reconstruction_weight"],
                "mask_ratio": summary["mask_ratio"],
                "clean_train_accuracy": clean.get("accuracy"),
                "best_validation_accuracy": summary["best_validation_accuracy"],
                "best_validation_macro_f1": summary["best_validation_macro_f1"],
                "best_epoch": summary["best_epoch"],
                "total_seconds": summary["total_seconds"],
                "test_evaluated": summary["test_evaluated"],
            }
        )
    if len(rows) != 8:
        raise RuntimeError(f"Expected 8 completed joint-MAE runs, found {len(rows)}")
    output = base / "comparison.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
