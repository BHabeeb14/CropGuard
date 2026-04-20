"""
Build hierarchical folder layout from combined_data (fine class folders).

Outputs:
  <out>/crop/<crop_key>/          — all images for that crop (any disease)
  <out>/disease/<crop_key>/<disease_key>/ — images for that crop+disease
  <out>/hierarchy_meta.json       — fine class → crop/disease keys for Flutter/training

Usage (from ml_backend):
  python build_hierarchical_dataset.py --from combined_data --out hierarchical_data
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from crop_hierarchy.label_parser import parse_fine_class_to_crop_disease, sanitize_fs_segment

_IMAGE_EXT = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


def _images_in_dir(d: str) -> list[str]:
    out = []
    try:
        for fn in os.listdir(d):
            if fn.lower().endswith(_IMAGE_EXT):
                out.append(os.path.join(d, fn))
    except OSError:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="combined_data", help="Fine-grained class folder root")
    ap.add_argument("--out", dest="out", default="hierarchical_data", help="Output root")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    out_root = os.path.abspath(args.out)
    crop_root = os.path.join(out_root, "crop")
    disease_root = os.path.join(out_root, "disease")

    if not os.path.isdir(src):
        print(f"Not found: {src}", file=sys.stderr)
        sys.exit(1)

    fine_dirs = sorted(
        d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)) and not d.startswith(".")
    )
    meta: dict = {"fine": {}, "crop_keys": [], "disease_trees": {}}
    crop_keys_set: set[str] = set()

    for fine in fine_dirs:
        crop_raw, disease_raw = parse_fine_class_to_crop_disease(fine)
        crop_key = sanitize_fs_segment(crop_raw)
        disease_key = sanitize_fs_segment(disease_raw)
        meta["fine"][fine] = {"crop_key": crop_key, "disease_key": disease_key}
        crop_keys_set.add(crop_key)

        src_class = os.path.join(src, fine)
        paths = _images_in_dir(src_class)
        if not paths:
            continue

        if args.dry_run:
            continue

        cdir = os.path.join(crop_root, crop_key)
        ddir = os.path.join(disease_root, crop_key, disease_key)
        os.makedirs(cdir, exist_ok=True)
        os.makedirs(ddir, exist_ok=True)

        for i, p in enumerate(paths):
            base = os.path.basename(p)
            c_dst = os.path.join(cdir, f"h_{crop_key}_{i:05d}_{base}")
            d_dst = os.path.join(ddir, f"h_{crop_key}_{i:05d}_{base}")
            shutil.copy2(p, c_dst)
            shutil.copy2(p, d_dst)

    # Summaries
    if args.dry_run:
        meta["crop_keys"] = sorted(crop_keys_set)
        for ck in meta["crop_keys"]:
            meta["disease_trees"][ck] = sorted(
                {meta["fine"][f]["disease_key"] for f in fine_dirs if meta["fine"][f]["crop_key"] == ck}
            )
        crops = meta["crop_keys"]
    else:
        crops = sorted(
            d for d in os.listdir(crop_root) if os.path.isdir(os.path.join(crop_root, d))
        )
        meta["crop_keys"] = crops
        for ck in crops:
            dp = os.path.join(disease_root, ck)
            if os.path.isdir(dp):
                meta["disease_trees"][ck] = sorted(
                    x for x in os.listdir(dp) if os.path.isdir(os.path.join(dp, x))
                )

    meta_path = os.path.join(out_root, "hierarchy_meta.json")
    os.makedirs(out_root, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")
    print(f"Crops: {len(crops)}")
    print(f"Output: {out_root}")
    print("Next: python train_hierarchical.py --data-dir hierarchical_data")


if __name__ == "__main__":
    main()
