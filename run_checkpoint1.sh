#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
FORCE=${FORCE:-0}

cd "$ROOT"

if [[ ! -d data/train ]]; then
  echo "ERROR: dataset not found at $ROOT/data/train" >&2
  exit 1
fi

echo "============================================================"
echo "Checkpoint 1: supervised controls"
echo "Models: ResNet-18 and Tiny ViT"
echo "Initialization: random; no external pretrained weights"
echo "Model selection: fixed validation split only"
echo "Labeled test and test2: not loaded"
echo "============================================================"

run_experiment() {
  local config_path=$1
  local run_dir=$2
  local checkpoint_path=$3

  if [[ -f "$run_dir/summary.json" && "$FORCE" != "1" ]]; then
    echo "Skipping completed run: $run_dir"
  else
    "$PYTHON_BIN" scripts/train_supervised.py \
      --config "$config_path" \
      --device auto
  fi

  if [[ -f "$run_dir/history.csv" ]]; then
    "$PYTHON_BIN" scripts/plot_history.py "$run_dir"
  fi

  if [[ -f "$checkpoint_path" ]]; then
    if [[ ! -f "$run_dir/clean_train_metrics.json" || "$FORCE" == "1" ]]; then
      "$PYTHON_BIN" scripts/evaluate_clean_train.py \
        --checkpoint "$checkpoint_path" \
        --output "$run_dir/clean_train_metrics.json" \
        --device auto
    fi
  fi
}

run_experiment \
  configs/checkpoint1_resnet18.yaml \
  results/checkpoint1/supervised_resnet18_seed0 \
  artifacts/checkpoints/checkpoint1/supervised_resnet18_seed0/best.pt

run_experiment \
  configs/checkpoint1_vit_tiny.yaml \
  results/checkpoint1/supervised_vit_tiny_seed0 \
  artifacts/checkpoints/checkpoint1/supervised_vit_tiny_seed0/best.pt

echo "Checkpoint 1 training finished. Review results/checkpoint1 before committing."

