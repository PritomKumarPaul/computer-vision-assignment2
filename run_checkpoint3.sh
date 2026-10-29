#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
FORCE=${FORCE:-0}
cd "$ROOT"
mkdir -p logs/checkpoint3

if [[ ! -d data/train ]]; then
  echo "ERROR: dataset not found at $ROOT/data/train" >&2
  exit 1
fi

run_joint_mae() {
  local config=$1
  local run_name=$2
  local result_dir="results/checkpoint3/$run_name"
  local checkpoint="artifacts/checkpoints/checkpoint3/$run_name/best.pt"

  if [[ -f "$result_dir/summary.json" && "$FORCE" != "1" ]]; then
    echo "Skipping completed joint-MAE run: $run_name"
  else
    "$PYTHON_BIN" scripts/train_joint_mae.py --config "$config" --device auto \
      2>&1 | tee "logs/checkpoint3/${run_name}.log"
  fi

  if [[ -f "$result_dir/history.csv" ]]; then
    "$PYTHON_BIN" scripts/plot_joint_mae_history.py "$result_dir"
  fi
  if [[ -f "$checkpoint" && ( ! -f "$result_dir/clean_train_metrics.json" || "$FORCE" == "1" ) ]]; then
    "$PYTHON_BIN" scripts/evaluate_clean_train.py \
      --checkpoint "$checkpoint" \
      --output "$result_dir/clean_train_metrics.json" \
      --device auto
  fi
}

echo "Checkpoint 3: joint classification and MAE-style reconstruction"
echo "Factorial sweep: loss 0.75/0.25 vs 0.50/0.50; masking 50% vs 75%"
echo "Model selection: clean validation classification only"
echo "Test and test2: not loaded"

run_joint_mae configs/checkpoint3_joint_mae_resnet18.yaml joint_mae_resnet18_seed0
run_joint_mae configs/checkpoint3_joint_mae_vit_tiny.yaml joint_mae_vit_tiny_seed0
run_joint_mae configs/checkpoint3_joint_mae_resnet18_cls75_mae25_mask75.yaml joint_mae_resnet18_cls75_mae25_mask75_seed0
run_joint_mae configs/checkpoint3_joint_mae_vit_tiny_cls75_mae25_mask75.yaml joint_mae_vit_tiny_cls75_mae25_mask75_seed0
run_joint_mae configs/checkpoint3_joint_mae_resnet18_cls50_mae50_mask50.yaml joint_mae_resnet18_cls50_mae50_mask50_seed0
run_joint_mae configs/checkpoint3_joint_mae_vit_tiny_cls50_mae50_mask50.yaml joint_mae_vit_tiny_cls50_mae50_mask50_seed0
run_joint_mae configs/checkpoint3_joint_mae_resnet18_cls50_mae50_mask75.yaml joint_mae_resnet18_cls50_mae50_mask75_seed0
run_joint_mae configs/checkpoint3_joint_mae_vit_tiny_cls50_mae50_mask75.yaml joint_mae_vit_tiny_cls50_mae50_mask75_seed0

"$PYTHON_BIN" scripts/summarize_checkpoint3.py
echo "Checkpoint 3 complete. Test and test2 were not evaluated."
