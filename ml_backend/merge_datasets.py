"""
Merge PlantVillage with the Kaggle dataset:
  nirmalsankalana/crop-pest-and-disease-detection

Run after: pip install kagglehub
Requires: ~/.kaggle/kaggle.json (Kaggle API credentials)

Output: a single folder of class subfolders suitable for train_model.py
  image_dataset_from_directory(combined_dir)

Merge rules:
- PlantVillage classes are copied as-is (folder names like Tomato___Late_blight).
- Extra dataset: if a class folder name exactly matches a PlantVillage class,
  images are added to that folder (more training data). Otherwise a new class
  is created as CPDD_<sanitized_name> to avoid accidental label collisions.

More crops / classes (50+ crop coverage goal):
- After this script, run merge_extended_sources.py with crop_datasets_manifest.json
  to append additional Kaggle sets as EXT_<prefix>_<class>/ folders (no label
  collision with PlantVillage). Then: python estimate_crop_coverage.py combined_data
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from collections import defaultdict

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")


def _is_image(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXT)


def collect_images_by_class(root: str) -> dict[str, list[str]]:
    """
    Build {class_name: [image paths]}.
    Supports: train/val/test splits, or flat class folders at root.
    """
    root = os.path.abspath(root)
    result: dict[str, list[str]] = defaultdict(list)
    if not os.path.isdir(root):
        return dict(result)

    split_names = ("train", "val", "valid", "test")
    has_split = any(os.path.isdir(os.path.join(root, s)) for s in split_names)

    if has_split:
        for split in split_names:
            sp = os.path.join(root, split)
            if not os.path.isdir(sp):
                continue
            for sub in os.listdir(sp):
                subpath = os.path.join(sp, sub)
                if not os.path.isdir(subpath):
                    continue
                for fn in os.listdir(subpath):
                    if _is_image(fn):
                        result[sub].append(os.path.join(subpath, fn))
        if result:
            return dict(result)

    # Flat: each subdirectory is one class
    for sub in os.listdir(root):
        subpath = os.path.join(root, sub)
        if not os.path.isdir(subpath) or sub.startswith("."):
            continue
        for fn in os.listdir(subpath):
            if _is_image(fn):
                result[sub].append(os.path.join(subpath, fn))

    return dict(result)


def sanitize_new_class(name: str) -> str:
    """Safe folder name for filesystem and class_labels.txt."""
    s = re.sub(r"[^\w\-]+", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def merge_to_output(
    plantvillage_root: str,
    extra_root: str,
    output_dir: str,
    *,
    dry_run: bool = False,
) -> list[str]:
    """
    Copy all images into output_dir with one subfolder per class.
    Returns sorted list of class folder names (for sanity check).
    """
    pv = collect_images_by_class(plantvillage_root)
    ex = collect_images_by_class(extra_root)

    if not pv:
        raise ValueError(f"No class folders with images under: {plantvillage_root}")
    if not ex:
        raise ValueError(f"No class folders with images under: {extra_root}")

    pv_classes = set(pv.keys())
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    # 1) Copy PlantVillage
    for cls, paths in pv.items():
        dest = os.path.join(output_dir, cls)
        if not dry_run:
            os.makedirs(dest, exist_ok=True)
        for i, src in enumerate(paths):
            base = os.path.basename(src)
            dst = os.path.join(dest, f"pv_{i:04d}_{base}")
            if not dry_run:
                shutil.copy2(src, dst)

    # 2) Merge extra: same class name -> same folder; else CPDD_<name>
    for cls, paths in ex.items():
        if cls in pv_classes:
            dest_dir = os.path.join(output_dir, cls)
        else:
            safe = sanitize_new_class(cls)
            new_name = f"CPDD_{safe}"
            dest_dir = os.path.join(output_dir, new_name)
        if not dry_run:
            os.makedirs(dest_dir, exist_ok=True)
        for i, src in enumerate(paths):
            base = os.path.basename(src)
            dst = os.path.join(dest_dir, f"cpdd_{i:04d}_{base}")
            if not dry_run:
                shutil.copy2(src, dst)

    if dry_run:
        extra_names = set()
        for c in ex:
            if c not in pv_classes:
                extra_names.add(f"CPDD_{sanitize_new_class(c)}")
        return sorted(pv_classes | extra_names)

    return sorted(d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)))


def merge_prefixed_extra_into_output(
    output_dir: str,
    extra_root: str,
    prefix: str,
    *,
    dry_run: bool = False,
) -> int:
    """
    Append images from a third-party dataset without merging into PlantVillage folder names.
    Each source class becomes: EXT_<prefix>_<sanitized_class>/
    This avoids collisions (e.g. two different datasets both using 'healthy').
    Returns number of image files copied (or that would be copied).
    """
    output_dir = os.path.abspath(output_dir)
    extra_root = find_best_extra_root(os.path.abspath(extra_root))
    by_class = collect_images_by_class(extra_root)
    if not by_class:
        raise ValueError(f"No class folders with images under: {extra_root}")

    safe_prefix = sanitize_new_class(prefix) or "src"
    copied = 0
    for cls, paths in by_class.items():
        safe_cls = sanitize_new_class(cls)
        dest_name = f"EXT_{safe_prefix}_{safe_cls}"
        dest_dir = os.path.join(output_dir, dest_name)
        if not dry_run:
            os.makedirs(dest_dir, exist_ok=True)
        for i, src in enumerate(paths):
            base = os.path.basename(src)
            dst = os.path.join(dest_dir, f"ext_{safe_prefix}_{i:05d}_{base}")
            if not dry_run:
                shutil.copy2(src, dst)
            copied += 1
    return copied


def download_plantvillage():
    import kagglehub

    return kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")


def download_crop_pest():
    import kagglehub

    return kagglehub.dataset_download("nirmalsankalana/crop-pest-and-disease-detection")


def find_plantvillage_image_root(dataset_path: str) -> str:
    """Reuse same discovery as train_model.find_image_root."""
    for root, dirs, _ in os.walk(dataset_path):
        class_like = [d for d in dirs if "___" in d or "healthy" in d.lower()]
        if len(class_like) >= 5:
            return root
        if "color" in dirs:
            color_path = os.path.join(root, "color")
            if os.path.isdir(color_path):
                subdirs = os.listdir(color_path)
                if any("___" in d for d in subdirs):
                    return color_path
    return dataset_path


def find_best_extra_root(dataset_path: str) -> str:
    """Pick directory with the most class-like subfolders containing images."""
    best_root = dataset_path
    best_count = 0
    for root, dirs, _ in os.walk(dataset_path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        n = 0
        for d in dirs:
            p = os.path.join(root, d)
            try:
                files = os.listdir(p)
            except OSError:
                continue
            if any(_is_image(f) for f in files[:20]):
                n += 1
        if n > best_count:
            best_count = n
            best_root = root
    return best_root


def main():
    parser = argparse.ArgumentParser(
        description="Merge PlantVillage + Crop Pest and Disease (Kaggle) into one training folder."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="combined_data",
        help="Output directory (default: combined_data)",
    )
    parser.add_argument(
        "--plantvillage",
        help="Path to PlantVillage root (if omitted, download via kagglehub)",
    )
    parser.add_argument(
        "--extra",
        help="Path to crop-pest-and-disease dataset root (if omitted, download via kagglehub)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only, do not copy files",
    )
    args = parser.parse_args()

    out = os.path.abspath(args.output)

    print("CropGuard — merge PlantVillage + Crop Pest and Disease")
    print("=" * 60)

    if args.plantvillage:
        pv_path = find_plantvillage_image_root(os.path.abspath(args.plantvillage))
        print(f"PlantVillage root: {pv_path}")
    else:
        print("Downloading PlantVillage (kagglehub)...")
        pv_path = find_plantvillage_image_root(download_plantvillage())
        print(f"PlantVillage root: {pv_path}")

    if args.extra:
        ex_path = find_best_extra_root(os.path.abspath(args.extra))
        print(f"Extra dataset root: {ex_path}")
    else:
        print("Downloading crop-pest-and-disease (kagglehub)...")
        ex_path = find_best_extra_root(download_crop_pest())
        print(f"Extra dataset root: {ex_path}")

    pv_classes = collect_images_by_class(pv_path)
    ex_classes = collect_images_by_class(ex_path)
    print(f"\nPlantVillage classes: {len(pv_classes)}")
    print(f"Extra dataset classes: {len(ex_classes)}")
    pv_imgs = sum(len(v) for v in pv_classes.values())
    ex_imgs = sum(len(v) for v in ex_classes.values())
    print(f"PlantVillage images: {pv_imgs}")
    print(f"Extra images: {ex_imgs}")

    if args.dry_run:
        names = merge_to_output(pv_path, ex_path, out, dry_run=True)
        print(f"\nDry run — would create {len(names)} classes (no files written).")
        return

    print(f"\nMerging into: {out}")
    classes = merge_to_output(pv_path, ex_path, out)
    print(f"Done. Total classes in output: {len(classes)}")
    print("\nNext step:")
    print(f"  python train_model.py --data-dir \"{out}\"")


if __name__ == "__main__":
    main()
