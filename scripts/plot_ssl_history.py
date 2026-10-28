#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("run_dir", type=Path); args = parser.parse_args()
    history = pd.read_csv(args.run_dir / "history.csv")
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(history["epoch"], history["ssl_loss"])
    axis.set(xlabel="Epoch", ylabel="SSL loss", title="Self-supervised pretraining")
    axis.grid(alpha=0.25); figure.tight_layout(); figure.savefig(args.run_dir / "ssl_loss_curve.png", dpi=180); plt.close(figure)


if __name__ == "__main__": main()

