#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "kagglehub==0.3.6",
# ]
# ///

"""
Download, extract, and validate the Sign Language Dataset for YOLOv7.

Usage:
    uv run data/download_dataset.py

The dataset is downloaded from Kaggle and extracted into data/.
Structure after extraction:
    data/
    ├── train/
    │   ├── images/  (*.jpg)
    │   └── labels/  (*.txt)
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
"""

import argparse
import shutil
import sys
from pathlib import Path

import kagglehub

DATASET_SLUG = "daskoushik/sign-language-dataset-for-yolov7"
DATA_DIR = Path(__file__).resolve().parent

def validate_structure(split: str) -> tuple[int, int, list[str]]:
    split_dir = DATA_DIR / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    errors: list[str] = []

    if not split_dir.exists():
        return 0, 0, [f"Missing split directory: {split_dir}"]

    if not images_dir.exists():
        errors.append(f"Missing images directory: {images_dir}")

    if not labels_dir.exists():
        errors.append(f"Missing labels directory: {labels_dir}")

    if not images_dir.exists() or not labels_dir.exists():
        return 0, 0, errors

    image_files = sorted(images_dir.glob("*.jpg"))
    label_files = sorted(labels_dir.glob("*.txt"))

    image_stems = {f.stem for f in image_files}
    label_stems = {f.stem for f in label_files}

    images_without_labels = image_stems - label_stems
    labels_without_images = label_stems - image_stems

    if images_without_labels:
        errors.append(
            f"  {len(images_without_labels)} images without labels: "
            f"{', '.join(sorted(images_without_labels)[:5])}..."
        )

    if labels_without_images:
        errors.append(
            f"  {len(labels_without_images)} labels without images: "
            f"{', '.join(sorted(labels_without_images)[:5])}..."
        )

    return len(image_files), len(label_files), errors


def download() -> Path:
    print(f"Downloading {DATASET_SLUG}...")
    cache_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    print(f"  Downloaded to cache: {cache_path}")
    return cache_path


def extract(cache_path: Path) -> None:
    print("Extracting to data/ ...")

    for item in cache_path.iterdir():
        dest = DATA_DIR / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            print(f"  Copied {item.name}/ → {dest}")
        elif item.suffix == ".zip":
            shutil.unpack_archive(str(item), DATA_DIR)
            print(f"  Unpacked {item.name} → {DATA_DIR}")


def report(split: str, n_images: int, n_labels: int, errors: list[str]) -> bool:
    print(f"{split}: {n_images} images, {n_labels} labels")
    ok = True
    for err in errors:
        print(f"[!] {err}")
        ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and validate the sign language dataset"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if data already exists",
    )
    args = parser.parse_args()

    splits = ["train", "valid", "test"]
    all_ok = True

    existing_splits = [s for s in splits if (DATA_DIR / s).exists()]

    if existing_splits and not args.force:
        print(
            "Dataset already exists. Use --force to re-download."
        )
        for split in existing_splits:
            n_imgs, n_lbls, errors = validate_structure(split)
            ok = report(split, n_imgs, n_lbls, errors)
            all_ok = all_ok and ok
        sys.exit(0 if all_ok else 1)

    if args.force:
        print("Force mode: removing existing data...")
        for split in splits:
            split_dir = DATA_DIR / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
                print(f"  Removed {split_dir}")

    cache_path = download()
    extract(cache_path)

    print("\nValidating structure...")
    print(f"  Root: {DATA_DIR}")
    for split in splits:
        n_imgs, n_lbls, errors = validate_structure(split)
        ok = report(split, n_imgs, n_lbls, errors)
        all_ok = all_ok and ok

    if all_ok:
        print(f"\n [OK] Dataset ready for training.")
    else:
        print(f"\n [ERROR] Dataset has issues. Check warnings above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
