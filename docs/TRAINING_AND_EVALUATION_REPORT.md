# CropGuard Edge AI — Training, Testing, and Evaluation Report

**MSc Dissertation: Design and Implementation of a Mobile Application with Machine Learning-Based Crop Disease Detection**

---

## 1. Dataset

### 1.1 Source and Description

The **PlantVillage** dataset was used for model training and evaluation. It was obtained via Kaggle (abdallahalidev/plantvillage-dataset) using the kagglehub Python library.

| Property | Value |
|----------|-------|
| **Total images** | 54,305 |
| **Number of classes** | 38 |
| **Crop types** | 14 (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato) |
| **Label format** | Categorical (one-hot encoded) |
| **Image format** | RGB colour images |

### 1.2 Data Split

A **70:30 train–validation split** was applied with a fixed random seed (42) for reproducibility:

| Split | Images | Purpose |
|-------|--------|---------|
| **Training** | 38,014 | Model training with augmentation |
| **Validation** | 16,291 | Validation and early stopping |

### 1.3 Data Augmentation

To improve robustness for real-world field conditions, the following augmentations were applied during training:

| Augmentation | Parameters | Rationale |
|--------------|------------|-----------|
| Random rotation | ±20% | Handles varying camera angles |
| Random horizontal flip | — | Leaf orientation invariance |
| Random brightness | ±20% | Different lighting conditions |
| Random contrast | ±20% | Shadows and overexposure |

Images were resized to **224×224** pixels to match the MobileNetV3 input size.

---

## 2. Model Architecture

### 2.1 Base Model

**MobileNetV3-Small** was used as the backbone, pre-trained on ImageNet. It was chosen for:

- **Efficiency:** Suitable for on-device inference on mobile hardware
- **Accuracy:** Strong performance on image classification
- **Size:** Small footprint for offline deployment

### 2.2 Architecture Summary

| Layer | Output shape | Parameters |
|-------|--------------|------------|
| Input | (224, 224, 3) | 0 |
| MobileNetV3-Small (ImageNet) | (576,) | 939,120 |
| Dropout (0.3) | (576,) | 0 |
| Dense (softmax) | (38,) | 21,926 |
| **Total** | — | **961,046** |
| **Trainable** | — | **948,934** |
| **Non-trainable** | — | **12,112** |

### 2.3 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1×10⁻³ |
| Loss function | Categorical cross-entropy |
| Batch size | 64 |
| Epochs | 2 |
| Fine-tuning | Full model (base + head) |

### 2.4 Callbacks

- **ModelCheckpoint:** Saves the best model based on validation accuracy
- **ReduceLROnPlateau:** Reduces learning rate by 0.5 when validation loss plateaus (patience: 3)
- **EarlyStopping:** Stops training if validation accuracy does not improve for 3 epochs (patience: 3)

---

## 3. Training Results

### 3.1 Training Progress

| Epoch | Train loss | Train accuracy | Val loss | Val accuracy |
|-------|------------|-----------------|----------|--------------|
| 1 | 0.3191 | 90.87% | 0.0795 | **97.41%** |
| 2 | 0.0758 | 97.57% | 0.0605 | **98.06%** |

### 3.2 Final Performance

| Metric | Value |
|--------|-------|
| **Best validation accuracy** | **98.06%** |
| **Best validation loss** | 0.0605 |
| **Final training accuracy** | 97.57% |

The model reached **98.06% validation accuracy** after 2 epochs, indicating strong generalisation on unseen data. The low validation loss (0.0605) and small gap between training and validation accuracy suggest minimal overfitting.

---

## 4. Testing and Validation

### 4.1 Validation Strategy

- The validation set (16,291 images) was held out and never used for training.
- Validation metrics were computed at the end of each epoch.
- The best model (highest validation accuracy) was saved and used for export.

### 4.2 Evaluation Metrics

- **Accuracy:** Primary metric for multi-class classification
- **Loss:** Categorical cross-entropy for training monitoring

### 4.3 Interpretation

A validation accuracy of **98.06%** on 38 classes indicates that the model:

- Generalises well to unseen images
- Handles the diversity of disease and healthy states across 14 crop types
- Benefits from transfer learning (ImageNet pre-training) and data augmentation

---

## 5. Model Export and Quantization

### 5.1 Export Pipeline

The trained Keras model was exported to **TensorFlow Lite (TFLite)** for deployment on mobile devices. The export process included:

1. **Representative dataset:** A subset of training images (up to 500 samples) was used for calibration.
2. **8-bit quantization:** Weights were quantized from float32 to int8 to reduce model size.
3. **Input/output:** Float32 input and output were retained for compatibility with the Flutter tflite_flutter package (images normalised to 0–1).

### 5.2 Quantization Results

| Property | Before (Keras) | After (TFLite) |
|----------|----------------|----------------|
| **Format** | float32 | int8 (weights), float32 (I/O) |
| **Size** | ~3.8 MB (approx.) | **1,218 KB** |
| **Size reduction** | — | **~75%** |

### 5.3 Export Outputs

| File | Description |
|------|-------------|
| `cropguard_model_quantized.tflite` | Quantized TFLite model for mobile inference |
| `class_labels.txt` | 38 class names (one per line) for mapping indices to labels |

---

## 6. Mobile Deployment

### 6.1 Integration

The TFLite model was integrated into the Flutter mobile app via the `tflite_flutter` package. The inference service:

- Loads the model from app assets at runtime
- Preprocesses camera-captured images to 224×224, normalised to [0, 1]
- Runs inference on-device (no network required)
- Returns top predictions with confidence scores

### 6.2 Platform Support

| Platform | TFLite support |
|----------|-----------------|
| Android | ✓ Full support |
| iOS | ✓ Full support |
| Web | ✗ Stub (no TFLite on web) |

---

## 7. Summary Table

| Aspect | Details |
|--------|---------|
| **Dataset** | PlantVillage, 54,305 images, 38 classes |
| **Train/val split** | 70% / 30% |
| **Model** | MobileNetV3-Small, transfer learning |
| **Best validation accuracy** | **98.06%** |
| **TFLite model size** | 1,218 KB |
| **Quantization** | 8-bit (weights) |
| **Deployment** | Flutter, offline-first |

---

## 8. Limitations

### 8.1 Limited Dataset Scope

A key limitation of this project is the **restricted scope of the training dataset**. The PlantVillage dataset covers only **14 crop types** and **38 disease/health classes** in total. As a result, the CropGuard application can reliably identify only the following crops and their associated conditions:

| Crop | Classes (disease states) |
|------|--------------------------|
| Apple | Apple scab, Black rot, Cedar apple rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery mildew, Healthy |
| Corn (maize) | Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy |
| Grape | Black rot, Esca, Leaf blight, Healthy |
| Orange | Huanglongbing (Citrus greening) |
| Peach | Bacterial spot, Healthy |
| Pepper (bell) | Bacterial spot, Healthy |
| Potato | Early blight, Late blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery mildew |
| Strawberry | Leaf scorch, Healthy |
| Tomato | Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy |

**Implications:**

- The model **cannot** recognise diseases or crops outside this set. Images of wheat, rice, cassava, banana, or any crop not listed above will be misclassified into one of the 38 classes, leading to incorrect or misleading results.
- The system is best suited to **smallholder contexts** where these 14 crops are commonly grown, rather than as a general-purpose crop disease detector.
- Expanding coverage would require **additional labelled datasets** and **retraining** the model.

This limitation is acknowledged as a constraint of the current implementation and is documented for transparency in the dissertation.

---

## 9. Conclusion

The CropGuard model achieved **98.06% validation accuracy** on the PlantVillage dataset, demonstrating strong performance for crop disease classification. The model was successfully quantized to TFLite (1,218 KB) and integrated into the Flutter mobile app for offline, on-device inference. The combination of MobileNetV3-Small, data augmentation, and transfer learning produced a compact, accurate model suitable for deployment in low-connectivity agricultural settings.
