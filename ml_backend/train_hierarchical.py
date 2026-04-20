"""
Train hierarchical models: (1) crop classifier, (2) per-crop disease classifiers.

Prerequisites:
  python build_hierarchical_dataset.py --from combined_data --out hierarchical_data

Usage (from ml_backend):
  python train_hierarchical.py --data-dir hierarchical_data --stage all
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import train_model as tm

OUTPUT_SUB = os.path.join(tm.OUTPUT_DIR, "hierarchical")
CROP_KERAS = os.path.join(OUTPUT_SUB, "cropguard_crop.keras")
CROP_TFLITE = os.path.join(OUTPUT_SUB, "cropguard_crop.tflite")
CROP_LABELS = os.path.join(OUTPUT_SUB, "crop_labels.txt")
MANIFEST_JSON = os.path.join(OUTPUT_SUB, "hierarchy_manifest.json")


def _ensure_dirs() -> None:
    os.makedirs(os.path.join(OUTPUT_SUB, "diseases"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_SUB, "labels", "diseases"), exist_ok=True)


def train_crop(data_root: str, epochs: int) -> list[str]:
    crop_dir = os.path.join(data_root, "crop")
    if not os.path.isdir(crop_dir):
        raise FileNotFoundError(f"Missing {crop_dir}. Run build_hierarchical_dataset.py first.")

    print("\n[Stage: crop] Loading dataset...")
    train_ds, val_ds, class_names = tm.load_dataset(crop_dir)
    n = len(class_names)
    print(f"      Crop classes: {n} — {class_names[:8]}...")

    _ensure_dirs()
    with open(CROP_LABELS, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    model = tm.build_model(n)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=tm.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        ModelCheckpoint(CROP_KERAS, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    model.load_weights(CROP_KERAS)
    tm.export_to_tflite(model, train_ds, CROP_TFLITE)
    print(f"      Saved {CROP_TFLITE}")
    return class_names


def train_diseases(data_root: str, crop_order: list[str], epochs: int) -> dict:
    disease_root = os.path.join(data_root, "disease")
    if not os.path.isdir(disease_root):
        raise FileNotFoundError(f"Missing {disease_root}")

    exports: dict[str, dict] = {}
    for crop_key in crop_order:
        path = os.path.join(disease_root, crop_key)
        if not os.path.isdir(path):
            print(f"      [skip] no folder for crop: {crop_key}")
            exports[crop_key] = {"mode": "single", "label": "unknown"}
            continue
        subs = sorted(
            d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and not d.startswith(".")
        )
        if len(subs) < 2:
            label = subs[0] if subs else "unknown"
            exports[crop_key] = {"mode": "single", "label": label}
            print(f"      [single] {crop_key} → only class '{label}' (no disease model)")
            continue

        print(f"\n      [disease] {crop_key} — {len(subs)} classes")
        train_ds, val_ds, class_names = tm.load_dataset(path)
        k = len(class_names)
        labels_path = os.path.join(OUTPUT_SUB, "labels", "diseases", f"{crop_key}.txt")
        with open(labels_path, "w", encoding="utf-8") as f:
            f.write("\n".join(class_names))

        keras_path = os.path.join(OUTPUT_SUB, "diseases", f"{crop_key}.keras")
        tflite_path = os.path.join(OUTPUT_SUB, "diseases", f"{crop_key}.tflite")
        model = tm.build_model(k)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=tm.LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks = [
            ModelCheckpoint(keras_path, monitor="val_accuracy", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        ]
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
        model.load_weights(keras_path)
        tm.export_to_tflite(model, train_ds, tflite_path)
        exports[crop_key] = {
            "mode": "model",
            "keras": keras_path,
            "tflite": tflite_path,
            "labels": labels_path,
            "class_names": class_names,
        }
        print(f"      Saved {tflite_path}")

    for ck in crop_order:
        if ck not in exports:
            exports[ck] = {"mode": "single", "label": "unknown"}

    return exports


def write_flutter_manifest(crop_order: list[str], exports: dict[str, dict]) -> None:
    """Paths relative to Flutter project assets/."""
    crops_out = []
    for crop_key in crop_order:
        ex = exports.get(crop_key, {"mode": "single", "label": "unknown"})
        if ex.get("mode") == "single":
            crops_out.append(
                {
                    "crop_key": crop_key,
                    "disease": {"mode": "single", "label": ex.get("label", "unknown")},
                }
            )
        else:
            crops_out.append(
                {
                    "crop_key": crop_key,
                    "disease": {
                        "mode": "model",
                        "model": f"assets/models/hierarchical/diseases/{crop_key}.tflite",
                        "labels": f"assets/labels/hierarchical/diseases/{crop_key}.txt",
                    },
                }
            )

    manifest = {
        "enabled": True,
        "crop_model": "assets/models/hierarchical/cropguard_crop.tflite",
        "crop_labels": "assets/labels/hierarchical/crop_labels.txt",
        "crops": crops_out,
    }
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {MANIFEST_JSON}")
    print("Copy output/hierarchical/* into mobile_app/assets/ as described in manifest.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="hierarchical_data", help="Output of build_hierarchical_dataset.py")
    ap.add_argument("--stage", choices=("crop", "disease", "all"), default="all")
    ap.add_argument("--epochs", type=int, default=None, help="Override EPOCHS from train_model")
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_dir)
    if args.epochs is not None:
        tm.EPOCHS = args.epochs

    crop_order: list[str] = []
    exports: dict[str, dict] = {}

    if args.stage in ("crop", "all"):
        crop_order = train_crop(data_root, tm.EPOCHS)
    elif args.stage == "disease":
        if not os.path.isfile(CROP_LABELS):
            raise SystemExit(f"Missing {CROP_LABELS}. Run --stage crop first.")
        with open(CROP_LABELS, encoding="utf-8") as f:
            crop_order = [ln.strip() for ln in f if ln.strip()]

    if args.stage in ("disease", "all"):
        if not crop_order:
            with open(CROP_LABELS, encoding="utf-8") as f:
                crop_order = [ln.strip() for ln in f if ln.strip()]
        exports = train_diseases(data_root, crop_order, tm.EPOCHS)
        write_flutter_manifest(crop_order, exports)


if __name__ == "__main__":
    main()
