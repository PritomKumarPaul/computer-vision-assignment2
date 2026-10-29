#!/usr/bin/env python
"""Plot component losses and classification metrics for joint MAE training."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    history = pd.read_csv(args.run_dir / "history.csv")
    with (args.run_dir / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    classification_weight = float(config["objective"]["classification_weight"])
    reconstruction_weight = float(config["objective"]["reconstruction_weight"])

    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.plot(
        history["epoch"], history["train_classification_loss"], label="Train CE"
    )
    axis.plot(
        history["epoch"],
        history["validation_classification_loss"],
        label="Validation CE",
    )
    axis.set(xlabel="Epoch", ylabel="Cross-entropy", title="Classification loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.run_dir / "classification_loss_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.plot(history["epoch"], history["train_mae_loss"], label="Raw MAE loss")
    axis.plot(
        history["epoch"],
        history["weighted_mae_loss"],
        label=f"{reconstruction_weight:.2f} × MAE loss",
    )
    axis.set(
        xlabel="Epoch",
        ylabel="Masked-patch MSE",
        title="Auxiliary reconstruction loss",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.run_dir / "mae_loss_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.plot(history["epoch"], history["train_total_loss"], label="Total")
    axis.plot(
        history["epoch"],
        history["weighted_classification_loss"],
        label=f"{classification_weight:.2f} × classification",
    )
    axis.plot(
        history["epoch"],
        history["weighted_mae_loss"],
        label=f"{reconstruction_weight:.2f} × MAE",
    )
    axis.set(xlabel="Epoch", ylabel="Weighted loss", title="Joint objective")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.run_dir / "joint_loss_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.plot(history["epoch"], history["train_accuracy"], label="Train accuracy")
    axis.plot(
        history["epoch"], history["validation_accuracy"], label="Validation accuracy"
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
