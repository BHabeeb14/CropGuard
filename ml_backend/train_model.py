"""
CropGuard Edge AI - Model Training Script
MSc Dissertation: Design and Implementation of a Mobile Application with 
Machine Learning-Based Crop Disease Detection

Trains a MobileNetV3 model on PlantVillage dataset with data augmentation
and exports to TensorFlow Lite with 8-bit quantization for offline mobile inference.
"""

import argparse
import hashlib
import logging
import os
from typing import Optional

# Quieter TF C++/Python logs (shuffle-buffer INFO, augmentation graph "while_loop" warnings).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

logging.getLogger("tensorflow").setLevel(logging.ERROR)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Match keras.utils.image_dataset.ALLOWLIST_FORMATS
_IMAGE_EXT_ALLOWLIST = (".bmp", ".gif", ".jpeg", ".jpg", ".png")

# =============================================================================
# CONFIGURATION
# =============================================================================
# Speed tips: ↑ BATCH_SIZE (64/128), ↓ EPOCHS, use GPU (tensorflow with CUDA),
# optional: cache dataset to disk with .cache("path") after loading

IMG_SIZE = (224, 224)
BATCH_SIZE = 64  # Larger = fewer batches/epoch (32→64 ~2x faster, needs more RAM)
EPOCHS = 2
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.30  # 70:30 train-test split
SEED = 42
OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "cropguard_model.keras")
TFLITE_PATH = os.path.join(OUTPUT_DIR, "cropguard_model_quantized.tflite")
LABELS_PATH = os.path.join(OUTPUT_DIR, "class_labels.txt")
# After ignore_errors(), many skipped corrupt files can make a large shuffle buffer
# very slow to fill; keep modest. Shuffle runs AFTER cache so epoch 1 fills cache sequentially.
# 256 fills faster on CPU than 512; still enough randomization for large datasets.
SHUFFLE_BUFFER = 256

# =============================================================================
# DATASET LOADING (PlantVillage via kagglehub)
# =============================================================================


def _get_kagglehub_cache_path():
    """Return cached PlantVillage path when offline (no internet)."""
    cache_base = os.path.expanduser(
        os.path.join("~", ".cache", "kagglehub", "datasets", "abdallahalidev", "plantvillage-dataset")
    )
    if not os.path.isdir(cache_base):
        return None
    versions_dir = os.path.join(cache_base, "versions")
    if not os.path.isdir(versions_dir):
        return cache_base
    versions = [d for d in os.listdir(versions_dir) if os.path.isdir(os.path.join(versions_dir, d))]
    if not versions:
        return cache_base
    return os.path.join(versions_dir, max(versions, key=lambda v: int(v) if v.isdigit() else 0))


def download_plantvillage_dataset():
    """Download PlantVillage dataset using kagglehub. Falls back to cache when offline."""
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is required. Install with: pip install kagglehub\n"
            "Ensure Kaggle API credentials are at ~/.kaggle/kaggle.json"
        )

    try:
        dataset_path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
        return dataset_path
    except Exception as e:
        cached = _get_kagglehub_cache_path()
        if cached and os.path.isdir(cached):
            print(f"      Offline: using cached dataset at {cached}")
            return cached
        raise RuntimeError(
            f"Failed to download dataset ({e}). "
            "No cached copy found. Run with internet first."
        ) from e


def find_image_root(dataset_path):
    """
    Locate the root directory containing class subfolders.
    PlantVillage structure varies: may be in raw/color/, PlantVillage/, etc.
    """
    for root, dirs, _ in os.walk(dataset_path):
        # Look for directories that look like class folders (e.g., Apple___Apple_scab)
        class_like = [d for d in dirs if "___" in d or "healthy" in d.lower()]
        if len(class_like) >= 5:  # PlantVillage has 38 classes
            return root
        # Also check for common structure: raw/color/
        if "color" in dirs:
            color_path = os.path.join(root, "color")
            if os.path.isdir(color_path):
                subdirs = os.listdir(color_path)
                if any("___" in d for d in subdirs):
                    return color_path
    return dataset_path


def _index_image_paths(image_root: str) -> tuple[list[str], list[int], list[str]]:
    """
    Same layout as Keras: sorted class folders, sorted filenames per class.
    """
    class_names = sorted(
        d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))
    )
    paths: list[str] = []
    labels: list[int] = []
    for i, name in enumerate(class_names):
        sub = os.path.join(image_root, name)
        for fn in sorted(os.listdir(sub)):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in _IMAGE_EXT_ALLOWLIST:
                continue
            paths.append(os.path.join(sub, fn))
            labels.append(i)
    return paths, labels, class_names


def _paths_to_tf_dataset(
    paths: list[str],
    labels: np.ndarray,
    num_classes: int,
    batch_size: int,
) -> tf.data.Dataset:
    """Decode/resize like image_dataset_from_directory (bilinear, categorical labels)."""

    def decode(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE, method="bilinear")
        img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
        one_hot = tf.one_hot(tf.cast(label, tf.int32), num_classes)
        return img, one_hot

    paths_arr = np.asarray(paths, dtype=object)
    labels_arr = np.asarray(labels, dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths_arr, labels_arr))
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds


def _skip_corrupt_images(dataset):
    """
    Drop samples that fail JPEG decode (merged datasets sometimes include corrupt files).
    Unbatch -> ignore_errors -> rebatch so one bad file does not abort the whole batch.
    """
    unbatched = dataset.unbatch()
    try:
        unbatched = unbatched.ignore_errors()
    except AttributeError:
        unbatched = unbatched.apply(tf.data.experimental.ignore_errors())
    return unbatched.batch(BATCH_SIZE, drop_remainder=False)


def load_dataset(data_dir, image_root: Optional[str] = None):
    """Load image dataset with stratified 70:30 train/val split and data augmentation."""
    # Find the correct image root (or use explicit root for hierarchical disease folders)
    image_root = image_root or find_image_root(data_dir)

    # Data augmentation for 'In-the-Wild' robustness
    # Rotation, brightness, horizontal flip as specified
    train_augmentation = keras.Sequential([
        layers.RandomRotation(0.2),  # ±20% rotation
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(0.2),  # ±20% brightness
        layers.RandomContrast(0.2),  # Additional contrast for field conditions
    ])

    paths, labels, class_names = _index_image_paths(image_root)
    if not paths:
        raise ValueError(f"No images found under {image_root}")
    num_classes = len(class_names)
    labels_arr = np.asarray(labels, dtype=np.int32)

    # Keras image_dataset_from_directory uses a GLOBAL file split (last 30% of all files).
    # Files are ordered by class folder, so validation can be almost disjoint from training
    # (e.g. mostly late alphabet classes) → high train acc, ~random val acc. Use stratified split.
    try:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths,
            labels_arr,
            test_size=VALIDATION_SPLIT,
            stratify=labels_arr,
            random_state=SEED,
        )
        print(
            f"      Train/val: stratified split (~{(1 - VALIDATION_SPLIT) * 100:.0f}% / "
            f"~{VALIDATION_SPLIT * 100:.0f}% per class)."
        )
    except ValueError:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths,
            labels_arr,
            test_size=VALIDATION_SPLIT,
            random_state=SEED,
        )
        print(
            "      Warning: stratified split not possible (some classes too small); "
            "using random shuffle split."
        )

    train_ds = _paths_to_tf_dataset(train_paths, train_labels, num_classes, BATCH_SIZE)
    val_ds = _paths_to_tf_dataset(val_paths, val_labels, num_classes, BATCH_SIZE)
    train_ds.class_names = class_names
    val_ds.class_names = class_names

    # Skip corrupt JPEG/PNG in batch (common in large merged downloads)
    train_ds = _skip_corrupt_images(train_ds)

    # Cache BEFORE shuffle: avoids holding a huge shuffle buffer while decoding/skippping
    # (otherwise "Filling shuffle buffer: 40 of 10000" can take a very long time).
    # _strat suffix: invalidates old caches from the pre-stratified global split.
    root_key = hashlib.md5(os.path.abspath(image_root).encode("utf-8")).hexdigest()[:12]
    cache_root = os.path.join(
        OUTPUT_DIR, "cache", f"{len(class_names)}classes_{root_key}_strat"
    )
    os.makedirs(cache_root, exist_ok=True)
    train_ds = train_ds.cache(os.path.join(cache_root, "train"))
    train_ds = train_ds.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)

    val_ds = _skip_corrupt_images(val_ds)
    val_ds = val_ds.cache(os.path.join(cache_root, "val"))

    # Ensure [0, 255] range (some loaders return [0,1]; model expects [0,255])
    def ensure_input_range(images, labels):
        return tf.cond(
            tf.reduce_max(images) <= 1.0,
            lambda: (images * 255.0, labels),
            lambda: (images, labels),
        )

    train_ds = train_ds.map(ensure_input_range, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(ensure_input_range, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply augmentation to training pipeline
    def augment_images(images, labels):
        return train_augmentation(images, training=True), labels

    train_ds = train_ds.map(augment_images, num_parallel_calls=tf.data.AUTOTUNE)

    # Optimize dataset pipeline
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


# =============================================================================
# MODEL BUILDING (MobileNetV3)
# =============================================================================


def build_model(num_classes):
    """Build MobileNetV3-Small with custom classification head for transfer learning."""
    base_model = keras.applications.MobileNetV3Small(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = True  # Fine-tune entire model

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    # Rescale [0,255] -> [-1,1] to match ImageNet/MobileNet expected input
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    x = base_model(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model


# =============================================================================
# TFLITE EXPORT (8-bit Quantization)
# =============================================================================


def representative_dataset_gen(dataset, num_samples=500):
    """Generator for representative dataset used in quantization."""
    for images, _ in dataset.take(1):
        for i in range(min(num_samples, images.shape[0])):
            yield [tf.cast(images[i : i + 1], tf.float32)]


def export_to_tflite(model, train_ds, output_path):
    """
    Export model to TensorFlow Lite with 8-bit quantization.
    Reduces model size by ~75% (float32 -> int8 weights).
    Uses float32 input/output for Flutter compatibility (normalize 0-1).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable 8-bit integer quantization (weights only; ~75% size reduction)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset for calibration
    def rep_data_gen():
        for batch_images, _ in train_ds.take(10):
            for i in range(min(50, batch_images.shape[0])):
                yield [tf.cast(batch_images[i : i + 1], tf.float32)]

    converter.representative_dataset = rep_data_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]
    # Keep float32 I/O for Flutter tflite_flutter (input: 0-1 normalized)

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    quantized_size = len(tflite_model)
    print(f"\n✓ TFLite model saved to {output_path}")
    print(f"  Quantized model size: {quantized_size / 1024:.1f} KB (~75% smaller)")


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================


def resolve_dataset_path(data_dir: Optional[str]) -> str:
    """
    If data_dir is set, use it (e.g. merged PlantVillage + crop-pest from merge_datasets.py).
    Otherwise download PlantVillage via kagglehub.
    """
    if data_dir:
        path = os.path.abspath(data_dir)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"--data-dir not found: {path}")
        return path
    return download_plantvillage_dataset()


def main(data_dir: Optional[str] = None):
    print("=" * 60)
    print("CropGuard Edge AI - Model Training")
    print("=" * 60)

    # 1. Dataset path (merged folder or PlantVillage only)
    if data_dir:
        print("\n[1/5] Using local dataset (--data-dir)...")
    else:
        print("\n[1/5] Downloading PlantVillage dataset via kagglehub...")
    dataset_path = resolve_dataset_path(data_dir)
    print(f"      Dataset path: {dataset_path}")

    # 2. Load data
    print("\n[2/5] Loading dataset with 70:30 train-test split...")
    train_ds, val_ds, class_names = load_dataset(dataset_path)
    num_classes = len(class_names)
    print(f"      Classes: {num_classes}")
    print(f"      Sample classes: {class_names[:5]}...")

    # Save class labels for Flutter app
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LABELS_PATH, "w") as f:
        f.write("\n".join(class_names))
    print(f"      Labels saved to {LABELS_PATH}")

    # 3. Build and compile model
    print("\n[3/5] Building MobileNetV3 model...")
    model = build_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 4. Train
    print("\n[4/5] Training model...")
    print(
        "      First epoch: building on-disk cache + filling shuffle buffer can take many minutes "
        "on CPU — wait until you see batch progress (e.g. 1/868). Do not stop the process early."
    )
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=3,  # Stop if no improvement for 3 epochs (fits 10-epoch runs)
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Load best weights
    model.load_weights(MODEL_PATH)
    _, val_acc = model.evaluate(val_ds)
    print(f"\n      Best validation accuracy: {val_acc * 100:.2f}%")

    # 5. Export to TFLite
    print("\n[5/5] Exporting to TensorFlow Lite (8-bit quantization)...")
    export_to_tflite(model, train_ds, TFLITE_PATH)

    print("\n" + "=" * 60)
    print("Training complete! Copy the following to your Flutter assets:")
    print(f"  - {TFLITE_PATH}")
    print(f"  - {LABELS_PATH}")
    print("=" * 60)


def export_only(dataset_path):
    """Export existing trained model to TFLite (no retraining)."""
    print("=" * 60)
    print("CropGuard - Export to TFLite (existing model)")
    print("=" * 60)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run full training first."
        )
    print("\n[1/3] Loading dataset for quantization calibration...")
    train_ds, _, class_names = load_dataset(dataset_path)
    num_classes = len(class_names)
    print(f"      Classes: {num_classes}")
    print("\n[2/3] Building model and loading trained weights...")
    model = build_model(num_classes)
    model.load_weights(MODEL_PATH)
    print("\n[3/3] Exporting to TensorFlow Lite...")
    export_to_tflite(model, train_ds, TFLITE_PATH)
    print("\n" + "=" * 60)
    print("Export complete! Copy to Flutter assets:")
    print(f"  - {TFLITE_PATH}")
    print(f"  - {LABELS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export existing trained model to TFLite (skip training)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Training images root (class subfolders). Use merged output from merge_datasets.py. "
        "If omitted, PlantVillage is downloaded via kagglehub.",
    )
    args = parser.parse_args()
    if args.export_only:
        dataset_path = resolve_dataset_path(args.data_dir)
        export_only(dataset_path)
    else:
        main(data_dir=args.data_dir)
