# Preprocessing Alignment: Training ↔ App

This document describes how input preprocessing is aligned between the ML training pipeline and the Flutter app to ensure accurate inference.

---

## Training Pipeline

### 1. Dataset Loading

- **Source:** `image_dataset_from_directory` (PlantVillage)
- **Raw range:** May be [0, 255] or [0, 1] depending on TensorFlow version
- **Normalisation:** `ensure_input_range` maps [0, 1] → [0, 255] if needed, so the model always receives [0, 255]

### 2. Model Input

- **Expected range:** [0, 255] (float32)
- **First layer:** `Rescaling(1/127.5, offset=-1)` converts [0, 255] → [-1, 1] for MobileNet
- **Formula:** `output = input / 127.5 - 1`

---

## App Preprocessing

### Camera Image → Model Input

1. **Decode** JPEG from camera
2. **Apply EXIF orientation** (`bakeOrientation`) so images are rotated correctly
3. **Resize** to 224×224 with linear interpolation (matches TensorFlow bilinear)
4. **Pixel range:** [0, 255] (configurable via `_inputRange0_255` in `camera_screen.dart`)

### Format

- **Shape:** [1, 224, 224, 3] (batch, height, width, channels)
- **Channels:** RGB order (R, G, B)
- **Dtype:** float64 (double) in Dart

---

## Alignment Checklist

| Step | Training | App |
|------|----------|-----|
| Input range | [0, 255] | [0, 255] ✓ |
| Image size | 224×224 | 224×224 ✓ |
| Channel order | RGB | RGB ✓ |
| Rescaling | [0,255]→[-1,1] in model | N/A (model handles it) |

---

## If Results Are Wrong

1. **Retrain** after any training pipeline changes (e.g. `ensure_input_range`, Rescaling)
2. **Try [0, 1] in app:** Set `_inputRange0_255 = false` in `camera_screen.dart` if your model was trained with [0, 1], then rebuild
3. **Check model:** Ensure the TFLite model in `assets/models/` was exported from the same training run

---

## Retraining Required

After changing `train_model.py` (e.g. adding `ensure_input_range` or Rescaling), you must:

```bash
cd ml_backend
python train_model.py
python train_model.py --export-only
# Copy output to mobile_app/assets/
```
