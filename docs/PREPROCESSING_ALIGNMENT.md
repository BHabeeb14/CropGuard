# Preprocessing Alignment: Training ↔ App

This document describes how input preprocessing is aligned between the ML training pipeline and the Flutter app so that **on-device accuracy matches the validation figures reported in the Evaluation chapter** (97.24% across 60 classes). Any mismatch between training-time and inference-time preprocessing is a classic cause of "train–serve skew" and can silently destroy field accuracy even when the model is correct.

---

## Training Pipeline

### 1. Dataset Loading

- **Source:** the merged crop-disease corpus produced by `ml_backend/merge_datasets.py`, read by `image_dataset_from_directory` in `ml_backend/train_model.py`.
- **Raw pixel range out of `image_dataset_from_directory`:** may be `[0, 255]` or `[0, 1]` depending on TensorFlow version.
- **Normalisation guard:** `ensure_input_range` in the training script maps `[0, 1] → [0, 255]` when needed, so the model always receives `[0, 255]` inputs regardless of TF version.

### 2. Model Input

- **Expected range:** `[0, 255]` (float32).
- **First layer inside the model:** `Rescaling(1/127.5, offset=-1)` maps `[0, 255] → [−1, 1]`, which matches the distribution MobileNetV3 was pre-trained on.
- **Formula:** `output = input / 127.5 − 1`.

Placing the rescaling **inside** the exported model means the on-device code can send raw `[0, 255]` pixels and does not need to replicate any normalisation in Dart.

---

## App Preprocessing

### Camera Image → Model Input

1. **Decode** the JPEG produced by the camera.
2. **Apply EXIF orientation** (`bakeOrientation`) so the leaf is upright, matching the training data.
3. **Resize** to 224 × 224 with linear interpolation (matches TensorFlow's bilinear resize closely enough for MobileNetV3).
4. **Pixel range:** `[0, 255]`, because the model's Rescaling layer does the rest.

Implemented in `mobile_app/lib/screens/camera_screen.dart` inside `_preprocessImage`. The behaviour is controlled by the constant `_inputRange0_255 = true`.

### Format

- **Shape:** `[1, 224, 224, 3]` (batch, height, width, channels).
- **Channels:** RGB order (R, G, B).
- **Dtype:** float64 (Dart `double`), which the `tflite_flutter` interpreter converts to float32 at the boundary.

---

## Alignment Checklist

| Step | Training | App |
|------|----------|-----|
| Input range | `[0, 255]` | `[0, 255]` ✓ |
| Image size | 224 × 224 | 224 × 224 ✓ |
| Channel order | RGB | RGB ✓ |
| EXIF handling | N/A (curated images) | `bakeOrientation` applied ✓ |
| Rescaling to `[−1, 1]` | Done **inside** the exported model | Not needed in app (model handles it) |

---

## If Results Look Wrong on Device

1. **Confirm you are running the latest exported model.** The `.tflite` file in `mobile_app/assets/models/` must come from the same training run as the class labels in `mobile_app/assets/labels/class_labels.txt` (60 entries, matching the softmax output width).
2. **Check the input range flag.** In `camera_screen.dart`, `_inputRange0_255` must be `true` for models with an internal Rescaling layer. Set it to `false` only if you deliberately trained a model that expects `[0, 1]` (no Rescaling layer).
3. **Re-verify EXIF.** If a specific phone returns rotated frames, make sure `bakeOrientation` is being called before `copyResize`.
4. **Re-export after any training-pipeline change.** Any edit to `ensure_input_range`, the `Rescaling` layer, or the class list must be followed by a fresh export and an asset copy.

---

## Retraining / Re-export Workflow

After changing `ml_backend/train_model.py` (for example, altering the normalisation guard or the Rescaling layer):

```bash
cd ml_backend
python train_model.py                 # train + export TFLite
# or, if weights already exist and only the artefact needs rebuilding:
python train_model.py --export-only
```

Then copy the artefacts over to the app so the bundled assets match the trained model:

```bash
cp output/cropguard_model_quantized.tflite ../mobile_app/assets/models/
cp output/class_labels.txt                  ../mobile_app/assets/labels/
```

Finally, rebuild the Flutter app (`flutter build apk` or `flutter run`) so the new assets are packaged into the APK.

> The optional hierarchical pipeline (`train_hierarchical.py`) writes into `mobile_app/assets/models/hierarchical/` and `mobile_app/assets/labels/hierarchical/` instead. Do not mix hierarchical and flat assets in the same build; the app picks its path based on the presence of `assets/hierarchy_manifest.json`.
