"""
Scan a class-folder image tree (e.g. combined_data/) and delete files that PIL or TensorFlow
cannot decode (TF catches JPEGs that pass PIL but fail libjpeg during training).

Usage:
  conda activate crop_ai
  pip install pillow tensorflow
  python prune_corrupt_images.py combined_data
"""

from __future__ import annotations

import argparse
import os
import sys

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")


def _require_tf_decode(path: str) -> None:
    """
    Some JPEGs pass PIL but fail TensorFlow/libjpeg ('Corrupt JPEG', 'premature end').
    If TensorFlow is installed, decode the same way training does.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return
    raw = tf.io.read_file(path)
    p = path.lower()
    if p.endswith((".jpg", ".jpeg")):
        tf.io.decode_jpeg(raw, channels=3)
    else:
        tf.io.decode_image(raw, channels=3, expand_animations=False)


def main():
    parser = argparse.ArgumentParser(description="Remove unreadable image files from dataset folders.")
    parser.add_argument("root", help="Dataset root (e.g. combined_data)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths only, do not delete",
    )
    parser.add_argument(
        "--no-tf",
        action="store_true",
        help="Skip TensorFlow decode check (PIL only; may miss TF/libjpeg failures)",
    )
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    try:
        from PIL import Image
    except ImportError:
        print("Install Pillow: pip install pillow", file=sys.stderr)
        sys.exit(1)

    removed = 0
    checked = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.lower().endswith(IMAGE_EXT):
                continue
            path = os.path.join(dirpath, name)
            checked += 1
            try:
                with Image.open(path) as im:
                    im.verify()
                # verify() closes; reopen for full decode check
                with Image.open(path) as im2:
                    im2.load()
                if not args.no_tf:
                    _require_tf_decode(path)
            except Exception as e:
                print(f"BAD: {path}\n      ({e})")
                if not args.dry_run:
                    try:
                        os.remove(path)
                        removed += 1
                    except OSError as oe:
                        print(f"      could not delete: {oe}", file=sys.stderr)

    print(f"\nChecked {checked} image files. Removed {removed} corrupt files." + (" (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
