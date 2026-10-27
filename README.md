# Assignment 2: Self-Supervised Visual Representation Learning

This repository studies visual representation learning with 2,400 labeled
training images from 16 scene classes. Every model is initialized from scratch
and trained only with the supplied training images. External pretrained weights
and external training images are not used.

The project contains two controlled streams:

- a CNN stream centered on ResNet-18, MoCo/BYOL-style pretraining, and
  consistency-regularized fine-tuning;
- a transformer stream centered on a small Vision Transformer, DINO-style
  self-distillation, and joint masked-image reconstruction plus classification.

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

