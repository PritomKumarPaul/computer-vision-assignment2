#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def exact_group_subset(groups: list[list[Path]], target: int) -> set[int]:
    """Return group indices whose combined image count is exactly target."""
    options: dict[int, list[int]] = {0: []}
    for index, group in enumerate(groups):
        for count, chosen in sorted(options.items(), reverse=True):
            new_count = count + len(group)
            if new_count <= target and new_count not in options:
                options[new_count] = chosen + [index]
    if target not in options:
        raise RuntimeError(f"Cannot construct an exact validation subset of {target} images")
    return set(options[target])


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = ["relative_path", "class_name", "label", "sha256"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["relative_path"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    args = parser.parse_args()

    classes = sorted(path.name for path in args.data_root.iterdir() if path.is_dir())
    if not classes:
        raise RuntimeError(f"No class directories found beneath {args.data_root}")

    rng = random.Random(args.seed)
    train_rows, val_rows = [], []
    duplicate_groups = []

    for label, class_name in enumerate(classes):
        class_dir = args.data_root / class_name
        paths = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        target = round(len(paths) * args.val_fraction)
        by_hash: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            by_hash[digest(path)].append(path)
        grouped = list(by_hash.items())
        rng.shuffle(grouped)
        selected = exact_group_subset([members for _, members in grouped], target)

        for index, (sha256, members) in enumerate(grouped):
            if len(members) > 1:
                duplicate_groups.append([str(path.relative_to(args.data_root)) for path in members])
            destination = val_rows if index in selected else train_rows
            for path in members:
                destination.append(
                    {
                        "relative_path": str(path.relative_to(args.data_root)),
                        "class_name": class_name,
                        "label": label,
                        "sha256": sha256,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_dir / "train.csv", train_rows)
    write_manifest(args.output_dir / "val.csv", val_rows)

    summary = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "class_names": classes,
        "train_images": len(train_rows),
        "validation_images": len(val_rows),
        "train_per_class": dict(Counter(row["class_name"] for row in train_rows)),
        "validation_per_class": dict(Counter(row["class_name"] for row in val_rows)),
        "exact_duplicate_groups": duplicate_groups,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
