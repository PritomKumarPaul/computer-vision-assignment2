#!/usr/bin/env python
import csv, json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    base = ROOT / "results" / "checkpoint2" / "classification"
    for summary_path in sorted(base.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        clean_path = summary_path.parent / "clean_train_metrics.json"
        clean = json.loads(clean_path.read_text()) if clean_path.exists() else {}
        rows.append({
            "run_name": summary["run_name"], "method": summary["method"],
            "backbone": summary["model"], "protocol": summary["classification_protocol"],
            "clean_train_accuracy": clean.get("accuracy"),
            "best_validation_accuracy": summary["best_validation_accuracy"],
            "best_validation_macro_f1": summary["best_validation_macro_f1"],
            "best_epoch": summary["best_epoch"], "total_seconds": summary["total_seconds"],
            "test_evaluated": summary["test_evaluated"],
        })
    if len(rows) != 8:
        raise RuntimeError(f"Expected 8 completed downstream runs, found {len(rows)}")
    output = ROOT / "results" / "checkpoint2" / "comparison.csv"; output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(output)


if __name__ == "__main__": main()
