# CropGuard Edge AI — Training, Testing, and Evaluation Report

**MSc Dissertation: Design and Implementation of a Mobile Application with Machine Learning-Based Crop Disease Detection**

---

## 1. Dataset

### 1.1 Source and Description

The model is trained on a **merged crop-disease image corpus** assembled by the project's data pipeline (`ml_backend/merge_datasets.py`) from two independently curated leaf-image sources. Class folders from the two sources are unified into a single `Crop___Condition` tree; non-matching folders from the secondary source are prefixed (`CPDD_*`) so labels stay unambiguous. Corrupt and truncated files are removed by `prune_corrupt_images.py` before training.

| Property | Value |
|----------|-------|
| **Total images (post-clean)** | ≈ 76,400 |
| **Number of classes** | 60 |
| **Crops covered** | 16 (Apple, Blueberry, Cashew, Cassava, Cherry, Corn/Maize, Grape, Orange, Peach, Pepper bell, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato) |
| **Label format** | Categorical (one-hot encoded) |
| **Image format** | RGB colour images |

### 1.2 Data Split

A **class-stratified 70% / 30% train–validation split** was applied with a fixed random seed (42) for reproducibility. Stratification preserves the per-class proportion in both splits and replaced an earlier non-stratified directory-level split that had produced a severe train/validation accuracy gap.

| Split | Images (approx.) | Purpose |
|-------|------------------|---------|
| **Training** | ≈ 53,480 | Gradient updates with augmentation |
| **Validation** | ≈ 22,920 | Held-out; drives checkpoint selection |

### 1.3 Data Augmentation

Applied only to the training stream; validation images are never augmented.

| Augmentation | Parameters | Rationale |
|--------------|------------|-----------|
| Random rotation | ±20% | Varying camera angles |
| Random horizontal flip | — | Leaf orientation invariance |
| Random brightness | ±20% | Different lighting conditions |
| Random contrast | ±20% | Shadows and overexposure |

Images are resized to **224 × 224** pixels to match the MobileNetV3-Small input.

---

## 2. Model Architecture

### 2.1 Base Model

**MobileNetV3-Small** is used as the backbone, pre-trained on ImageNet. It is chosen for:

- **Efficiency:** suitable for on-device inference on mid-range Android hardware.
- **Accuracy:** strong performance on image classification benchmarks.
- **Size:** small footprint for offline distribution.

### 2.2 Architecture Summary

| Layer | Output shape | Purpose |
|-------|--------------|---------|
| Input | (224, 224, 3) | RGB in [0, 255] |
| Rescaling (1/127.5, offset −1) | (224, 224, 3) | Maps [0, 255] → [−1, 1] |
| MobileNetV3-Small (ImageNet) | (576,) | Feature extractor |
| Dropout (0.3) | (576,) | Regularisation |
| Dense (softmax) | (60,) | Class probabilities |

Placing the rescaling inside the model keeps the preprocessing aligned with the on-device pipeline (the app sends raw [0, 255] pixels); see `PREPROCESSING_ALIGNMENT.md`.

### 2.3 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | Adam |
| Learning rate | 1 × 10⁻³ |
| Loss function | Categorical cross-entropy |
| Batch size | 64 |
| Epochs | 2 (sufficient for a strong baseline) |
| Fine-tuning | Full model (base + head) |

### 2.4 Callbacks

- **ModelCheckpoint** — saves the best weights by validation accuracy.
- **ReduceLROnPlateau** — halves the learning rate when validation loss plateaus (patience 3).
- **EarlyStopping** — stops training if validation accuracy does not improve (patience 3).

---

## 3. Training Results

### 3.1 Training Progress

| Epoch | Train loss | Train accuracy | Val loss | Val accuracy |
|-------|------------|-----------------|----------|--------------|
| 1 | 0.4027 | 88.42% | 0.1412 | 95.63% |
| 2 | 0.1183 | 96.31% | **0.0894** | **97.24%** |

### 3.2 Final Performance

| Metric | Value |
|--------|-------|
| **Best validation accuracy** | **97.24%** |
| **Best validation loss** | 0.0894 |
| **Final training accuracy** | 96.31% |

A validation accuracy of **97.24%** across 60 classes, together with the small train/validation gap, indicates strong generalisation within the covered distribution. Validation accuracy slightly exceeding training accuracy in epoch 1 is expected under augmentation and stochastic batching.

A separate `model.evaluate` pass on the validation dataset after reloading the best weights confirms consistency with the best-epoch metrics within numerical tolerance.

---

## 4. Testing and Validation

### 4.1 Validation Strategy

- The validation set (≈ 22,920 images) is held out and never used for gradient updates.
- Validation metrics are computed at the end of each epoch.
- The best model (highest validation accuracy) is saved and used for export.

### 4.2 Evaluation Metrics

- **Accuracy** — primary metric for multi-class classification.
- **Loss** — categorical cross-entropy for training monitoring and calibration.

### 4.3 Interpretation

A validation accuracy of **97.24%** across 60 classes indicates that the MobileNetV3-Small backbone, with transfer learning and light augmentation, captures most of the visual variability in the merged corpus. Misclassifications, though small in aggregate, cluster between visually similar classes (for example, different leaf-spot syndromes on the same crop, or early-stage blight versus healthy foliage under low contrast).

---

## 5. Model Export and Quantisation

### 5.1 Export Pipeline

The trained Keras model is exported to **TensorFlow Lite (TFLite)** for on-device inference:

1. **Representative dataset** — up to 500 samples drawn from the training pipeline are used for calibration so quantisation scales reflect real activation ranges.
2. **8-bit post-training quantisation** — weights are quantised from float32 to int8 to reduce file size.
3. **Input / output** — float32 tensors are retained at the API boundary for compatibility with the `tflite_flutter` plugin; the model still consumes raw [0, 255] pixels and the internal Rescaling layer handles normalisation.

### 5.2 Quantisation Results

| Property | Before (Keras) | After (TFLite) |
|----------|----------------|----------------|
| **Format** | float32 | int8 (weights), float32 (I/O) |
| **Size** | ≈ 3.8 MB | **≈ 1.2 MB** (1,260,384 bytes) |
| **Size reduction** | — | **≈ 67%** |

### 5.3 Export Outputs

| File | Description |
|------|-------------|
| `cropguard_model_quantized.tflite` | Quantised TFLite model |
| `class_labels.txt` | 60 class names, one per line, aligned with softmax output indices |

---

## 6. Mobile Deployment

### 6.1 Integration

The TFLite model is integrated into the Flutter mobile app via the `tflite_flutter` package. At runtime the app:

- Loads the model and `class_labels.txt` from the asset bundle.
- Preprocesses camera-captured images to 224 × 224, EXIF-corrected, RGB in [0, 255].
- Runs inference on-device; no network request is made.
- Returns top predictions with confidence scores.
- Applies a confidence threshold so that low-confidence predictions are surfaced as uncertain.
- Looks up a structured advice record for the predicted class from a bundled JSON asset (`assets/advice/advice.json`), so that disease severity is reported independently of model confidence.

### 6.2 Platform Support

| Platform | TFLite support |
|----------|----------------|
| Android | ✓ Full support |
| iOS | ✓ Full support |
| Web | ✗ Stub (no TFLite on web) |

### 6.3 Optional Two-Stage Routing

A hierarchical variant is available for taxonomy growth: `train_hierarchical.py` produces a crop classifier plus one per-crop disease head. The app routes `crop → disease` when the `assets/hierarchy_manifest.json` and companion model assets are present. The flat 60-class model remains the default baseline.

---

## 7. Summary Table

| Aspect | Details |
|--------|---------|
| **Dataset** | Merged crop-disease corpus, ≈ 76,400 images, 60 classes |
| **Crops** | 16 |
| **Train / val split** | 70% / 30%, class-stratified, seed 42 |
| **Model** | MobileNetV3-Small, transfer learning |
| **Best validation accuracy** | **97.24%** |
| **TFLite model size** | ≈ 1.2 MB |
| **Quantisation** | 8-bit weights, float32 I/O |
| **Deployment** | Flutter, offline-first |
| **Advice layer** | JSON-backed, 60/60 classes covered |

---

## 8. Limitations

### 8.1 Taxonomy Scope

CropGuard can reliably identify only the following crops and conditions. Crops or diseases outside this set are out-of-distribution and may be misclassified even with high apparent confidence unless the confidence threshold or the optional hierarchical routing intercepts them.

| # | Crop | Classes | Conditions the model has a class for |
|---|------|--------:|--------------------------------------|
| 1  | Apple                | 4  | Apple scab; Black rot; Cedar apple rust; Healthy |
| 2  | Blueberry            | 1  | Healthy |
| 3  | Cashew               | 5  | Anthracnose; Gummosis; Leaf miner; Red rust; Healthy |
| 4  | Cassava              | 5  | Bacterial blight; Brown spot; Green mite; Mosaic; Healthy |
| 5  | Cherry (incl. sour)  | 2  | Powdery mildew; Healthy |
| 6  | Corn / Maize         | 10 | Cercospora (Gray) leaf spot; Common rust; Northern leaf blight; Fall armyworm; Grasshopper; Leaf beetle; Leaf blight; Leaf spot; Streak virus; Healthy |
| 7  | Grape                | 4  | Black rot; Esca (Black measles); Leaf blight (Isariopsis); Healthy |
| 8  | Orange               | 1  | Huanglongbing (Citrus greening) |
| 9  | Peach                | 2  | Bacterial spot; Healthy |
| 10 | Pepper, bell         | 2  | Bacterial spot; Healthy |
| 11 | Potato               | 3  | Early blight; Late blight; Healthy |
| 12 | Raspberry            | 1  | Healthy |
| 13 | Soybean              | 1  | Healthy |
| 14 | Squash               | 1  | Powdery mildew |
| 15 | Strawberry           | 2  | Leaf scorch; Healthy |
| 16 | Tomato               | 13 | Bacterial spot; Early blight; Late blight; Leaf mold; Septoria leaf spot; Spider mites (two-spotted); Target spot; Yellow leaf curl virus; Mosaic virus; Leaf blight; Leaf curl; Verticillium wilt; Healthy |
|    | **Total**            | **60** | **16 crops** |

Coverage is uneven: Tomato, Corn/Maize, Cashew, and Cassava each have five or more classes, whereas Blueberry, Orange, Raspberry, Soybean, and Squash contribute only one class apiece and so offer little within-crop discrimination.

### 8.2 Other Limitations

- **Single hold-out split.** Metrics come from one stratified split; a separate test set or k-fold protocol would strengthen statistical claims.
- **Corpus composition.** Per-class image counts are uneven; stratified splitting mitigates but does not eliminate long-tail effects.
- **Leaves only.** All training images depict leaves; photographs of fruits, stems, or whole plants are out-of-distribution.
- **Curated vs field imagery.** Results apply to the merged corpus distribution. Field photographs with heavy blur, glare, occlusion, or non-leaf backgrounds may perform worse.
- **Advice layer.** Textual recommendations are rule-based and are not evaluated by quantitative metrics in this report.

---

## 9. Conclusion

The CropGuard model reached **97.24% validation accuracy** on a class-stratified 70 / 30 split of the merged 60-class corpus, and exports to an ≈ **1.2 MB** TFLite artefact suitable for mid-range Android hardware. Integrated with a Flutter client, an EXIF-aware preprocessing path, a confidence threshold, and a JSON-backed advice layer covering all 60 classes, the system supports the thesis that a compact, transfer-learned classifier can provide usable offline crop-disease screening within a defined taxonomy — while its limits outside that taxonomy remain the main honest caveat.
