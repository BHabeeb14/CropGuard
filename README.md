# CropGuard Edge AI

**Design and Implementation of a Mobile Application with Machine Learning-Based Crop Disease Detection**

MSc Dissertation Project — an offline-first Edge AI solution for smallholder farmers.

---

## Project Aim

CropGuard delivers **real-time crop-disease screening** on mobile devices with no internet connectivity. The system combines:

- **MobileNetV3-Small** transfer learning on a **merged multi-source crop-leaf corpus** (60 fine-grained `Crop___Condition` classes spanning 16 crops).
- **TensorFlow Lite** with 8-bit post-training quantisation (~1.2 MB deployable artefact, ~97% validation accuracy).
- **Flutter** cross-platform app with camera-based scanning, confidence-thresholded results, colour-coded severity bands, and offline advice text.
- **Optional hierarchical routing** (crop-classifier then per-crop disease-classifier) for extending coverage without retraining the whole network.

Target users: smallholder farmers in low-connectivity regions who need immediate, actionable diagnostics. **No account required — open and scan.**

---

## Repository Structure

```
CropGuard-EdgeAI-Mobile/
├── ml_backend/                             # Python training + dataset pipeline
│   ├── merge_datasets.py                   # Merge source datasets into combined_data/
│   ├── merge_extended_sources.py           # Merge additional manifest-listed datasets
│   ├── crop_datasets_manifest.json         # Config: which extra datasets to merge
│   ├── prune_corrupt_images.py             # Remove corrupt JPEG/PNG files
│   ├── estimate_crop_coverage.py           # Report how many crops are covered
│   ├── train_model.py                      # Flat classifier training + TFLite export
│   ├── train_hierarchical.py               # Hierarchical training (crop + per-crop disease)
│   ├── build_hierarchical_dataset.py       # Restructure combined_data/ for hierarchical mode
│   ├── crop_hierarchy/                     # Label parsing + metadata for hierarchical mode
│   ├── requirements.txt
│   └── output/                             # Generated artefacts (after training)
│       ├── cropguard_model_quantized.tflite
│       ├── cropguard_model.keras
│       ├── class_labels.txt
│       └── hierarchical/                   # Per-crop models (when train_hierarchical.py is used)
│
├── mobile_app/                             # Flutter application
│   ├── lib/
│   │   ├── main.dart                       # App entry point
│   │   ├── theme/app_theme.dart            # Material 3 palette + typography
│   │   ├── screens/                        # Onboarding, Home, Camera, Result
│   │   └── services/                       # InferenceService (flat + hierarchical), AdviceService
│   ├── assets/
│   │   ├── hierarchy_manifest.json         # Flat vs hierarchical routing switch
│   │   ├── models/                         # TFLite model(s)
│   │   └── labels/                         # class_labels.txt + per-crop disease labels
│   ├── pubspec.yaml
│   └── ...
│
├── docs/                                   # Docs + figures
│   ├── images/                             # Thesis screenshots (onboarding, home, etc.)
│   ├── DISSERTATION_STATUS_REPORT.txt
│   ├── PREPROCESSING_ALIGNMENT.md
│   └── TRAINING_AND_EVALUATION_REPORT.md
│
├── README.md
└── .gitignore
```
---

## Setup Instructions

### Prerequisites

- **Python 3.9+** (Anaconda/Miniconda recommended)
- **Flutter SDK 3.16+**
- **Kaggle account** (for downloading source datasets)
- A mid-range Android device for realistic inference timing (camera is required)

---

### 1. Python environment (ML backend)

```bash
conda create -n crop_ai python=3.9 -y
conda activate crop_ai

cd ml_backend
pip install -r requirements.txt
```

Configure the Kaggle API (needed by `kagglehub` for automatic dataset download):

1. Go to https://www.kaggle.com/settings and create an API token — this downloads `kaggle.json`.
2. Place `kaggle.json` at:
   - **Windows:** `C:\Users\<user>\.kaggle\kaggle.json`
   - **macOS / Linux:** `~/.kaggle/kaggle.json`

---

### 2. Build the merged training corpus

```bash
cd ml_backend

# Download + merge the primary sources into combined_data/
python merge_datasets.py

# Optionally append additional sources listed in crop_datasets_manifest.json
python merge_extended_sources.py

# Remove corrupt JPEG/PNG files that would break the TensorFlow decoder
python prune_corrupt_images.py combined_data

# (optional) Report the number of distinct crop buckets
python estimate_crop_coverage.py combined_data
```

After this step, `ml_backend/combined_data/` contains per-class folders such as `Apple___Apple_scab/`, `Tomato___healthy/`, `CPDD_Cassava_mosaic/`, etc.

---

### 3. Train the flat classifier

```bash
conda activate crop_ai
cd ml_backend
python train_model.py --data-dir combined_data
```

**Training pipeline:**

- Class-stratified 70:30 train / validation split with seed `42`.
- Data augmentation: random rotation (±20%), horizontal flip, brightness (±20%), contrast (±20%).
- MobileNetV3-Small backbone with ImageNet weights, fine-tuned end-to-end.
- Post-training 8-bit quantisation using a representative training subset.
- Exports:
  - `output/cropguard_model.keras` (full-precision checkpoint)
  - `output/cropguard_model_quantized.tflite` (~1.2 MB, mobile-ready)
  - `output/class_labels.txt`

On a typical i5 laptop with 16 GB RAM, two epochs take roughly 20–40 minutes on CPU.

---

### 4. (Optional) Train the hierarchical models

```bash
# Restructure combined_data/ into hierarchical_data/
python build_hierarchical_dataset.py

# Train the crop classifier and all per-crop disease classifiers
python train_hierarchical.py --data-dir hierarchical_data --stage all
```

Outputs land in `ml_backend/output/hierarchical/` with a `hierarchy_manifest.json` the Flutter app can consume.

---

### 5. Copy artefacts into the Flutter app

```bash
# Windows PowerShell / cmd
copy ml_backend\output\cropguard_model_quantized.tflite mobile_app\assets\models\
copy ml_backend\output\class_labels.txt mobile_app\assets\labels\

# For hierarchical mode also copy:
#   ml_backend\output\hierarchical\*.tflite   -> mobile_app\assets\models\hierarchical\
#   ml_backend\output\hierarchical\*.txt      -> mobile_app\assets\labels\hierarchical\diseases\
#   ml_backend\output\hierarchical\hierarchy_manifest.json -> mobile_app\assets\hierarchy_manifest.json
```

*(On macOS / Linux use `cp` and forward slashes.)*

---

### 6. Run the Flutter app

```bash
cd mobile_app
flutter pub get
flutter run              # requires a connected Android device for the camera
```

---

## Technical Specifications

| Component              | Value                                                             |
|------------------------|-------------------------------------------------------------------|
| Model backbone         | MobileNetV3-Small (ImageNet pre-trained, fully fine-tuned)        |
| Training corpus        | Merged multi-source crop-leaf collection                          |
| Classes / crops        | **60 fine-grained classes across 16 crops**                       |
| Train / val split      | Class-stratified **70 / 30**, seed 42                             |
| Augmentation           | Rotation, horizontal flip, brightness, contrast                   |
| Quantisation           | 8-bit post-training quantisation (float I/O, int8 weights)        |
| TFLite artefact size   | **~1.2 MB**                                                       |
| Validation accuracy    | **~97% on the merged corpus** (see `docs/TRAINING_AND_EVALUATION_REPORT.md`) |
| Mobile framework       | Flutter (Material 3) + `tflite_flutter` + `camera`                |

---

## Crops Currently Recognised (16)

Apple, Blueberry, Cashew, Cassava, Cherry, Corn / Maize, Grape, Orange, Peach, Bell pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.

Full per-crop disease list lives in `ml_backend/output/class_labels.txt`. Anything outside this set is treated as **out of distribution** and surfaced to the user as an "Uncertain" prediction by the confidence-threshold gate.

---

## HCI Design

- **Onboarding carousel** (4 panels) explains purpose, capture tips, uncertainty, and severity bands — shown only on first launch (via `shared_preferences`).
- **Severity colour coding:** green (healthy), amber (low), orange (medium), red (high).
- **Confidence threshold:** low-confidence predictions show an explicit "Uncertain" state rather than a forced class label.
- **Poppins typography** via `google_fonts` for outdoor legibility.
- **Offline advice:** short heuristic recommendations derived from the predicted label, no remote API calls.

---

## License

Academic use — MSc Dissertation project.

---

## References

- TensorFlow Lite: https://www.tensorflow.org/lite
- Flutter: https://docs.flutter.dev
- MobileNetV3 (Howard et al., 2019): https://arxiv.org/abs/1905.02244
- PlantVillage dataset (one of the sources of the merged corpus): https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- Crop Pest and Disease Detection dataset (another source of the merged corpus): https://www.kaggle.com/datasets/nirmalsankalana/crop-pest-and-disease-detection
