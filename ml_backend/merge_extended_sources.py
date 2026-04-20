"""
Append extra Kaggle datasets into combined_data/ using crop_datasets_manifest.json.

Run AFTER baseline merge:
  python merge_datasets.py -o combined_data
  python merge_extended_sources.py --combined combined_data

Requires: pip install kagglehub, ~/.kaggle/kaggle.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from merge_datasets import (
    find_best_extra_root,
    merge_prefixed_extra_into_output,
)


def _load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge additional Kaggle sources (EXT_<prefix>_<class>) into combined_data."
    )
    parser.add_argument(
        "--combined",
        "-c",
        default="combined_data",
        help="Path to existing combined_data folder (default: combined_data)",
    )
    parser.add_argument(
        "--manifest",
        "-m",
        default=os.path.join(_SCRIPT_DIR, "crop_datasets_manifest.json"),
        help="JSON manifest of Kaggle sources",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not copy files")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N enabled sources (0 = all)",
    )
    args = parser.parse_args()

    combined = os.path.abspath(args.combined)
    if not os.path.isdir(combined):
        print(f"Not a directory: {combined}", file=sys.stderr)
        print("Run first: python merge_datasets.py -o combined_data", file=sys.stderr)
        sys.exit(1)

    try:
        import kagglehub
    except ImportError:
        print("Install kagglehub: pip install kagglehub", file=sys.stderr)
        sys.exit(1)

    data = _load_manifest(args.manifest)
    sources = data.get("sources", [])
    log_entries: list[dict] = []
    processed = 0
    skipped = 0

    for src in sources:
        if not src.get("enabled"):
            continue
        kid = src.get("kaggle_id")
        if not kid:
            skipped += 1
            continue
        if args.limit and processed >= args.limit:
            break

        prefix = src.get("prefix") or "src"
        crop = src.get("crop", "")
        print(f"\n--- {kid}  (prefix={prefix}, crop={crop}) ---")
        try:
            path = kagglehub.dataset_download(kid)
            root = find_best_extra_root(path)
            n = merge_prefixed_extra_into_output(
                combined, root, prefix, dry_run=args.dry_run
            )
            print(f"    OK — copied {n} images" if not args.dry_run else f"    dry-run — would copy {n} images")
            log_entries.append(
                {
                    "kaggle_id": kid,
                    "prefix": prefix,
                    "crop": crop,
                    "images": n,
                    "root_used": root,
                    "status": "ok",
                }
            )
        except Exception as e:
            print(f"    SKIP — {e}")
            traceback.print_exc()
            log_entries.append(
                {
                    "kaggle_id": kid,
                    "prefix": prefix,
                    "crop": crop,
                    "status": "error",
                    "error": str(e),
                }
            )
        processed += 1

    log_path = os.path.join(combined, "_extended_merge_log.json")
    if not args.dry_run:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"sources": log_entries}, f, indent=2)
        print(f"\nWrote log: {log_path}")

    print(f"\nDone. Processed {processed} enabled sources. (Skipped {skipped} disabled / null-id rows.)")
    print("Next: python train_model.py --data-dir combined_data")
    print("Then copy output/class_labels.txt and output/cropguard_model_quantized.tflite to mobile_app/assets/")


if __name__ == "__main__":
    main()
