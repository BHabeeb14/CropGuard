"""
Rough estimate of how many distinct 'crops' (plant species / product groups) appear
in combined_data class folder names. Not botanical taxonomy — for project planning.

Usage (from ml_backend):
  python estimate_crop_coverage.py combined_data
"""

from __future__ import annotations

import argparse
import os
import re
import sys


def infer_crop_from_class_name(name: str) -> str:
    """Map one training class folder name to a coarse crop bucket."""
    if "___" in name:
        return name.split("___", 1)[0].strip()
    if name.startswith("CPDD_"):
        m = re.match(r"^CPDD_([^_]+)_(.+)$", name)
        if m:
            return f"CPDD:{m.group(1)}"
        return f"CPDD:{name[5:40]}"
    if name.startswith("EXT_"):
        m = re.match(r"^EXT_([^_]+)_([^_]+)_", name)
        if m:
            return f"EXT:{m.group(1)}_{m.group(2)}"
        m2 = re.match(r"^EXT_([^_]+)_", name)
        if m2:
            return f"EXT:{m2.group(1)}"
    return name[:48]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("combined_data", nargs="?", default="combined_data")
    args = parser.parse_args()
    root = os.path.abspath(args.combined_data)
    if not os.path.isdir(root):
        print(f"Not found: {root}", file=sys.stderr)
        sys.exit(1)

    classes = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    )
    crops = {infer_crop_from_class_name(c) for c in classes}
    print(f"Class folders (model labels): {len(classes)}")
    print(f"Estimated distinct crop buckets: {len(crops)}")
    print("\nBuckets (sorted):")
    for x in sorted(crops):
        print(f"  - {x}")


if __name__ == "__main__":
    main()
