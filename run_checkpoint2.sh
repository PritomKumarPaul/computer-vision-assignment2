#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
FORCE=${FORCE:-0}
cd "$ROOT"
mkdir -p logs/checkpoint2

if [[ ! -d data/train ]]; then
  echo "ERROR: dataset not found at $ROOT/data/train" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/make_checkpoint2_downstream_configs.py

run_pretraining() {
  local config=$1
  local run_name=$2
  local result_dir="results/checkpoint2/pretraining/$run_name"
  if [[ -f "$result_dir/summary.json" && "$FORCE" != "1" ]]; then
    echo "Skipping completed pretraining: $run_name"
  else
    "$PYTHON_BIN" scripts/pretrain_ssl.py --config "$config" --device auto \
      2>&1 | tee "logs/checkpoint2/${run_name}.log"
  fi
  if [[ -f "$result_dir/history.csv" ]]; then
    "$PYTHON_BIN" scripts/plot_ssl_history.py "$result_dir"
  fi
}

run_downstream() {
  local run_name=$1
  local config="outputs/checkpoint2_configs/${run_name}.yaml"
  local result_dir="results/checkpoint2/classification/$run_name"
  local checkpoint="artifacts/checkpoints/checkpoint2/classification/$run_name/best.pt"
  if [[ -f "$result_dir/summary.json" && "$FORCE" != "1" ]]; then
    echo "Skipping completed classification: $run_name"
  else
    "$PYTHON_BIN" scripts/train_supervised.py --config "$config" --device auto \
      2>&1 | tee "logs/checkpoint2/${run_name}.log"
  fi
  if [[ -f "$result_dir/history.csv" ]]; then
    "$PYTHON_BIN" scripts/plot_history.py "$result_dir"
  fi
  if [[ -f "$checkpoint" && ( ! -f "$result_dir/clean_train_metrics.json" || "$FORCE" == "1" ) ]]; then
    "$PYTHON_BIN" scripts/evaluate_clean_train.py \
      --checkpoint "$checkpoint" \
      --output "$result_dir/clean_train_metrics.json" \
      --device auto
  fi
}

echo "Stage 1/2: four fixed-budget SSL pretraining runs"
run_pretraining configs/checkpoint2_moco_resnet18.yaml moco_resnet18_seed0
run_pretraining configs/checkpoint2_moco_vit_tiny.yaml moco_vit_tiny_seed0
run_pretraining configs/checkpoint2_dino_resnet18.yaml dino_resnet18_seed0
run_pretraining configs/checkpoint2_dino_vit_tiny.yaml dino_vit_tiny_seed0

echo "Visualizing the four frozen SSL feature spaces on validation data"
if [[ "$FORCE" == "1" ]]; then
  "$PYTHON_BIN" scripts/visualize_ssl_features.py --device auto --force
else
  "$PYTHON_BIN" scripts/visualize_ssl_features.py --device auto
fi

echo "Stage 2/2: eight independent classification runs"
for method in moco dino; do
  for backbone in resnet18 vit_tiny; do
    run_downstream "${method}_${backbone}_linear_probe_seed0"
    run_downstream "${method}_${backbone}_full_finetune_seed0"
  done
done

"$PYTHON_BIN" scripts/summarize_checkpoint2.py
echo "Checkpoint 2 complete. Test and test2 were not evaluated."
