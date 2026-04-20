# CropGuard Edge AI

**Design and Implementation of a Mobile Application with Machine Learning-Based Crop Disease Detection**

MSc Dissertation Project — An offline-first Edge AI solution for smallholder farmers.

---

## Project Aim

CropGuard delivers **real-time crop disease detection** on mobile devices without requiring internet connectivity. The system combines:

- **MobileNetV3** transfer learning on the PlantVillage dataset
- **TensorFlow Lite** with 8-bit quantization for efficient on-device inference
- **Flutter** mobile app with camera-based scanning and localized advice

Target users: smallholder farmers in low-connectivity regions who need immediate, actionable disease diagnostics. No account required — open and scan.

---

## Professional Folder Structure

```
CropGuard-EdgeAI-Mobile/
├── ml_backend/                    # Python ML training pipeline
│   ├── train_model.py             # Model training & TFLite export
│   ├── requirements.txt          # Python dependencies
│   └── output/                   # Generated model & labels (after training)
│       ├── cropguard_model_quantized.tflite
│       ├── cropguard_model.keras
│       └── class_labels.txt
│
├── mobile_app/                    # Flutter application
│   ├── lib/
│   │   ├── main.dart              # App entry point
│   │   ├── theme/                 # HCI design (color-coded indicators)
│   │   ├── screens/               # Home, Camera
│   │   ├── services/              # Inference, Advice
│   ├── assets/
│   │   ├── models/               # TFLite model (copy from ml_backend/output)
│   │   └── labels/               # class_labels.txt
│   ├── pubspec.yaml
│   └── ...
│
├── README.md
└── .gitignore
```

---

## Setup Instructions

### Prerequisites

- **Anaconda** (or Miniconda) — Python 3.9+
- **Flutter SDK** 3.16+
- **Kaggle account** — for PlantVillage dataset access

---

### 1. Anaconda Environment (ML Backend)

```bash
# Create and activate environment (TensorFlow, pandas, matplotlib, scikit-learn, Jupyter)
conda create -n crop_ai python=3.9 tensorflow pandas matplotlib scikit-learn jupyter -y
conda activate crop_ai

# Option A: Download via kagglehub (requires Kaggle API)
pip install kagglehub

# Option B: Manual download (recommended - faster)
# 1. Go to https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
# 2. Click Download (requires Kaggle login)
# 3. Unzip to ml_backend/plantvillage-dataset (or set LOCAL_DATASET_PATH in train_model.py)

# Navigate to ML backend
cd ml_backend

# Configure Kaggle API (only if using kagglehub - Option A)
# 1. Go to https://www.kaggle.com/settings
# 2. Create API token (downloads kaggle.json)
# 3. Place kaggle.json at:
#    - Windows: C:\Users\<user>\.kaggle\kaggle.json
#    - macOS/Linux: ~/.kaggle/kaggle.json
```

### 2. Train the Model

```bash
conda activate crop_ai
cd ml_backend
python train_model.py
```

**Training pipeline:**
- Downloads PlantVillage via `kagglehub`
- 70:30 train–test split
- Data augmentation: rotation (±20°), horizontal flip, brightness (±20%)
- MobileNetV3-Small with ImageNet weights, fine-tuned
- Exports to TFLite with 8-bit quantization (~75% size reduction)

**Output:** `output/cropguard_model_quantized.tflite` and `output/class_labels.txt`

### 3. Copy Model to Flutter Assets

```bash
# From project root
copy ml_backend\output\cropguard_model_quantized.tflite mobile_app\assets\models\
copy ml_backend\output\class_labels.txt mobile_app\assets\labels\
```

*(Use `cp` on macOS/Linux.)*

---

### 4. Flutter App Setup

```bash
cd mobile_app

# Install dependencies
flutter pub get

# Run on device (camera requires physical device)
flutter run
```

---

## Technical Specifications

| Component | Technology |
|-----------|------------|
| Model | MobileNetV3-Small |
| Dataset | PlantVillage (38 classes, 14 crops) |
| Data split | 70% train, 30% test |
| Augmentation | Rotation, flip, brightness |
| Export | TensorFlow Lite, 8-bit quantization |
| Mobile | Flutter, tflite_flutter, camera |

### ML Environment (`crop_ai`) — Libraries

| Library | Purpose |
|---------|---------|
| Python 3.9 | Runtime |
| TensorFlow | Model training, Keras, TFLite export |
| pandas | Data handling |
| matplotlib | Visualization |
| scikit-learn | Metrics & utilities |
| Jupyter | Interactive notebooks |
| kagglehub | PlantVillage dataset download |

---

## HCI Design

- **Color-coded severity:** Green (healthy), Amber (low), Orange (medium), Red (high)
- **Localized advice:** Offline recommendations per disease type
- **Poppins typography** for readability

---

## License

Academic use — MSc Dissertation project.

---

## References

- PlantVillage Dataset: [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- TensorFlow Lite: [Documentation](https://www.tensorflow.org/lite)
- Flutter: [Documentation](https://docs.flutter.dev)
