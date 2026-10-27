# Assignment 2: Self-Supervised Visual Representation Learning

This repository studies visual representation learning with 2,400 labeled
training images from 16 scene classes. Every model is initialized from scratch
and trained only with the supplied training images. External pretrained weights
and external training images are not used.

## Data

The dataset is deliberately excluded from Git. Place it at:

```text
data/
  train/<class_name>/*.jpg   # 2,400 labeled images
  test/<class_name>/*.jpg    # 400 labeled images; final evaluation only
  test2/*.jpg                # 400 unlabeled images; never used for training
```

The fixed split contains 1,920 training and 480 validation images, balanced
across all 16 classes. Exact SHA-256 duplicate groups are kept entirely within
one partition.

Run the read-only data audit and split tests with:

```bash
python scripts/audit_dataset.py --data-root data
python -m unittest discover -s tests -v
```

To regenerate the fixed split:

```bash
python scripts/create_split.py \
  --data-root data/train \
  --output-dir splits \
  --seed 0 \
  --val-fraction 0.20
```

## Reproducibility policy

- Model selection uses only the fixed validation partition.
- The labeled test set is evaluated only after a method and configuration are
  frozen.
- `test2` is not used for SSL, pseudo-labeling, model selection, or training.
- Each run will save its resolved configuration, seed, history, environment,
  checkpoint checksum, and diagnostic measurements.
- Large model weights will be hosted as a GitHub Release asset or through an
  external checkpoint link.

Training commands and final results will be added as their corresponding
checkpoint milestones are completed. The final two-page report is maintained
outside this repository and submitted separately through Canvas.

## Checkpoint 1: supervised controls

The first controlled comparison trains a randomly initialized ResNet-18 and a
randomly initialized small Vision Transformer. Both use the same fixed split,
224x224 inputs, scene-preserving augmentation, label smoothing, and
validation-only model selection. Their optimizer settings are recorded in the
YAML configurations.

Run both controls sequentially with:

```bash
conda activate vlm_clean
bash run_checkpoint1.sh
```

The runner skips experiments that already have a completed `summary.json`, so
it is safe to resume after one model finishes. Set `FORCE=1` only when an
intentional full retraining is required.

Or run one model at a time:

```bash
python scripts/train_supervised.py \
  --config configs/checkpoint1_resnet18.yaml \
  --device auto

python scripts/train_supervised.py \
  --config configs/checkpoint1_vit_tiny.yaml \
  --device auto
```

These commands do not load or evaluate the labeled test set or `test2`.

