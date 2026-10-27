#!/usr/bin/env python
"""Read-only integrity and exact-duplicate audit for the assignment data."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_paths(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_split(path: Path):
    modes = Counter()
    sizes = Counter()
    broken = []
    hashes = defaultdict(list)
    paths = image_paths(path)
    for image_path in paths:
        try:
            with Image.open(image_path) as image:
                modes[image.mode] += 1
                sizes[image.size] += 1
                image.verify()
            hashes[sha256(image_path)].append(image_path)
        except Exception as error:  # report every bad input rather than aborting
            broken.append((image_path, error))
    return paths, modes, sizes, broken, hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    audits = {}
    for split in ("train", "test", "test2"):
        path = args.data_root / split
        paths, modes, sizes, broken, hashes = inspect_split(path)
        audits[split] = hashes
        duplicate_groups = [members for members in hashes.values() if len(members) > 1]
        print(f"\n{split}: {len(paths)} images")
        print(f"  modes: {dict(modes)}")
        print(f"  unique sizes: {len(sizes)}")
        print(f"  broken images: {len(broken)}")
        print(f"  exact duplicate groups: {len(duplicate_groups)}")
        for members in duplicate_groups:
            print("   - " + " | ".join(str(item) for item in members))

    for left, right in (("train", "test"), ("train", "test2"), ("test", "test2")):
        overlap = set(audits[left]).intersection(audits[right])
        print(f"\nExact hash overlap {left} <-> {right}: {len(overlap)}")
        for value in sorted(overlap):
            print("  left:  " + " | ".join(str(item) for item in audits[left][value]))
            print("  right: " + " | ".join(str(item) for item in audits[right][value]))


if __name__ == "__main__":
    main()
