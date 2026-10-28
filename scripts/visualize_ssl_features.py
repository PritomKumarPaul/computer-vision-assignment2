#!/usr/bin/env python
"""Visualize frozen SSL validation features without using either test split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import ManifestImageDataset  # noqa: E402
from src.data.manifest_dataset import build_transforms  # noqa: E402
from src.models import build_encoder  # noqa: E402
from src.training import extract_features  # noqa: E402


CONFIG_PATHS = (
    "configs/checkpoint2_moco_resnet18.yaml",
    "configs/checkpoint2_dino_resnet18.yaml",
    "configs/checkpoint2_moco_vit_tiny.yaml",
    "configs/checkpoint2_dino_vit_tiny.yaml",
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def neighborhood_purity(features: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Fraction of the k nearest validation neighbors sharing the sample label."""

    similarities = features @ features.T
    similarities.fill_diagonal_(float("-inf"))
    neighbors = similarities.topk(min(k, len(features) - 1), dim=1).indices
    return float((labels[neighbors] == labels[:, None]).float().mean().item())


def make_embedding(features: torch.Tensor, labels: torch.Tensor, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    normalized = F.normalize(features.float(), dim=1).numpy()
    label_array = labels.numpy()

    pca_2d_model = PCA(n_components=2, svd_solver="full")
    pca_2d = pca_2d_model.fit_transform(normalized)

    intermediate_components = min(50, normalized.shape[0] - 1, normalized.shape[1])
    intermediate = PCA(
        n_components=intermediate_components,
        svd_solver="full",
    ).fit_transform(normalized)
    tsne_2d = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        max_iter=1_000,
        random_state=seed,
        metric="euclidean",
    ).fit_transform(intermediate)

    metrics = {
        "cosine_silhouette": float(
            silhouette_score(normalized, label_array, metric="cosine")
        ),
        "neighborhood_purity_at_5": neighborhood_purity(
            torch.from_numpy(normalized), labels, k=5
        ),
        "pca_2d_explained_variance": float(
            pca_2d_model.explained_variance_ratio_.sum()
        ),
        "tsne_trustworthiness_at_5": float(
            trustworthiness(intermediate, tsne_2d, n_neighbors=5)
        ),
    }
    return pca_2d, tsne_2d, metrics


def scatter_embedding(
    axis,
    embedding: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    title: str,
) -> None:
    colors = plt.get_cmap("tab20", len(class_names))
    for label, class_name in enumerate(class_names):
        mask = labels == label
        axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=20,
            alpha=0.78,
            color=colors(label),
            edgecolors="none",
            label=class_name,
        )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_alpha(0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda")
    )
    parser.add_argument(
        "--output",
        default="results/checkpoint2/feature_visualizations",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = resolve(args.output)
    combined_path = output_dir / "ssl_feature_embeddings.png"
    metrics_path = output_dir / "embedding_metrics.csv"
    expected_individual = [
        output_dir / f"{Path(path).stem.removeprefix('checkpoint2_')}_features.png"
        for path in CONFIG_PATHS
    ]
    if (
        not args.force
        and combined_path.exists()
        and metrics_path.exists()
        and all(path.exists() for path in expected_individual)
    ):
        print(f"Skipping completed SSL feature visualization: {output_dir}")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    class_names: list[str] | None = None
    labels_reference: np.ndarray | None = None

    for config_path in CONFIG_PATHS:
        with resolve(config_path).open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        checkpoint_path = (
            resolve(config["checkpoint_root"]) / config["run_name"] / "encoder.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing completed SSL encoder checkpoint: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        encoder, _ = build_encoder(
            checkpoint["backbone"], image_size=int(checkpoint["image_size"])
        )
        encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
        encoder.to(device)

        _, clean_transform = build_transforms(
            int(config["image_size"]),
            config["data"]["normalization_mean"],
            config["data"]["normalization_std"],
        )
        dataset = ManifestImageDataset(
            resolve(config["data"]["train_root"]),
            resolve(config["data"]["validation_manifest"]),
            clean_transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
            pin_memory=use_cuda,
        )
        features, labels = extract_features(
            encoder, loader, device, f"Features: {config['run_name']}"
        )
        pca_2d, tsne_2d, metrics = make_embedding(features, labels, args.seed)

        labels_numpy = labels.numpy()
        if class_names is None:
            class_names = dataset.class_names
            labels_reference = labels_numpy
        elif dataset.class_names != class_names or not np.array_equal(
            labels_numpy, labels_reference
        ):
            raise RuntimeError("Validation samples or class mappings differ across runs")

        result = {
            "run_name": config["run_name"],
            "method": config["method"],
            "backbone": config["backbone"],
            "validation_samples": len(dataset),
            **metrics,
            "pca": pca_2d,
            "tsne": tsne_2d,
            "labels": labels_numpy,
        }
        results.append(result)
        del encoder, features, checkpoint
        if use_cuda:
            torch.cuda.empty_cache()

    assert class_names is not None

    figure, axes = plt.subplots(len(results), 2, figsize=(15, 19))
    for row, result in enumerate(results):
        name = f"{result['method'].upper()} + {result['backbone']}"
        scatter_embedding(
            axes[row, 0],
            result["pca"],
            result["labels"],
            class_names,
            f"{name} — PCA\nexplained variance={result['pca_2d_explained_variance']:.3f}",
        )
        scatter_embedding(
            axes[row, 1],
            result["tsne"],
            result["labels"],
            class_names,
            f"{name} — t-SNE\nsilhouette={result['cosine_silhouette']:.3f}, purity@5={result['neighborhood_purity_at_5']:.3f}",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=8,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        "Frozen SSL Features on the Fixed Validation Split",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0.065, 1, 0.975))
    figure.savefig(combined_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    for result, output_path in zip(results, expected_individual):
        figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
        name = f"{result['method'].upper()} + {result['backbone']}"
        scatter_embedding(
            axes[0], result["pca"], result["labels"], class_names, f"{name} — PCA"
        )
        scatter_embedding(
            axes[1],
            result["tsne"],
            result["labels"],
            class_names,
            f"{name} — t-SNE",
        )
        handles, legend_labels = axes[1].get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            loc="lower center",
            ncol=8,
            frameon=False,
            fontsize=8,
        )
        figure.suptitle(
            f"silhouette={result['cosine_silhouette']:.3f}  |  "
            f"purity@5={result['neighborhood_purity_at_5']:.3f}  |  "
            f"t-SNE trustworthiness@5={result['tsne_trustworthiness_at_5']:.3f}",
            fontsize=11,
        )
        figure.tight_layout(rect=(0, 0.13, 1, 0.94))
        figure.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(figure)

    metric_fields = [
        "run_name",
        "method",
        "backbone",
        "validation_samples",
        "cosine_silhouette",
        "neighborhood_purity_at_5",
        "pca_2d_explained_variance",
        "tsne_trustworthiness_at_5",
        "test_evaluated",
    ]
    temporary_metrics = metrics_path.with_suffix(".csv.tmp")
    with temporary_metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    key: result[key]
                    for key in metric_fields
                    if key not in {"test_evaluated"}
                }
                | {"test_evaluated": False}
            )
    temporary_metrics.replace(metrics_path)

    metadata = {
        "split": "validation",
        "validation_manifest": "splits/val.csv",
        "test_evaluated": False,
        "random_seed": args.seed,
        "tsne_perplexity": 30,
        "tsne_iterations": 1_000,
        "feature_normalization": "L2",
        "class_names": class_names,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {combined_path.relative_to(ROOT)}")
    print(f"Wrote {metrics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
