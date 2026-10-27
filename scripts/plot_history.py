#!/usr/bin/env python
"""Create compact loss and accuracy/F1 plots from a completed history CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    history_path = args.run_dir / "history.csv"
    history = pd.read_csv(history_path)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(history["epoch"], history["train_loss"], label="Train")
    axis.plot(history["epoch"], history["validation_loss"], label="Validation")
    axis.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.run_dir / "loss_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(history["epoch"], history["train_accuracy"], label="Train accuracy")
    axis.plot(
        history["epoch"],
        history["validation_accuracy"],
        label="Validation accuracy",
    )
    axis.plot(
        history["epoch"],
        history["validation_macro_f1"],
        label="Validation macro-F1",
        linestyle="--",
    )
    axis.set(xlabel="Epoch", ylabel="Score", title="Classification performance")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.run_dir / "accuracy_f1_curve.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()

