# EcoSort AI – Intelligent Waste Classification and Segregation System Using Deep Learning

> *"Smart Waste Recognition. Responsible Disposal."*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Executive Summary & Objective

**EcoSort AI** is an end-to-end computer vision and waste segregation decision-support system designed to automate municipal solid waste sorting at the point of disposal. Utilizing deep Convolutional Neural Networks (CNNs), EcoSort AI classifies uploaded images into six canonical waste categories and couples predictions with a decoupled two-dustbin recommendation engine, providing actionable, step-by-step guidance to ensure source segregation compliance and maximize recyclable material recovery.

### Key Capabilities
- **Multi-Class Vision Classification:** High-precision discrimination across six waste streams: **Cardboard, Glass, Metal, Paper, Plastic, and Trash**.
- **Comparative Deep Learning Architectures:** Direct comparative evaluation of three independent models:
  1. **Custom CNN:** Built from scratch with 4 convolutional stages, Batch Normalization, and Dropout regularization.
  2. **ResNet-18:** ImageNet-pretrained backbone with residual skip connections and fine-tuned classification head.
  3. **DenseNet-121:** ImageNet-pretrained backbone with dense connectivity, iterative feature reuse, and fine-tuned classifier.
- **Two-Dustbin Segregation Logic:** Automated mapping to **Blue Dustbin (Dry Waste)** with specialized **User Verification Required** inspection protocols for residual **Trash**.
- **Decoupled Recommendation Engine:** Clear step-by-step physical handling instructions and environmental impact metrics for every category.
- **Modern Streamlit Application:** Professional UI built with a forest-green theme, interactive Plotly visualizations, confidence distributions, and academic model benchmarks.

---

## 2. Dataset Architecture & Preprocessing

The system utilizes the curated **TrashBusters/combined** dataset from HuggingFace, consisting of **29,849 images** distributed across three predefined splits.

### Dataset Overview
- **Total Published Images:** 29,849
- **Image Resolution:** Standardized to 224 × 224 RGB
- **Classes (6):** Cardboard, Glass, Metal, Paper, Plastic, Trash
- **Splits Preserved:**
  - **Training Split:** 20,894 images (~70%)
  - **Validation Split:** 4,477 images (~15%)
  - **Test Split:** 4,478 images (~15%)

### Preprocessing Pipeline (`src/preprocessing.py`)
1. **Parquet Ingestion:** Incrementally extracts image byte streams and labels from all 5 Parquet files without saturating system RAM.
2. **Integrity & Verification:** Performs PIL image verification (`im.verify()`) to filter any truncated or unreadable image payloads.
3. **Duplicate Detection:** Employs cryptographic SHA-256 byte hashing to detect duplicate sample instances across the corpus.
4. **Resolution Normalization:** Converts palette/RGBA modes into standardized 3-channel RGB and resizes to 224 × 224 via bilinear interpolation.
5. **Class Imbalance Mitigation:** Computes balanced class weights via inverse frequency ($w_c = \frac{N}{K \cdot N_c}$), stored in `processed_dataset/class_weights.json` and fed into PyTorch's `CrossEntropyLoss`.
6. **Data Augmentation (`src/dataset_loader.py`):**
   - Strictly applied **only** to the training split: Random horizontal flips ($p=0.5$), random rotations ($\pm 15^\circ$), color jitter (brightness, contrast, saturation $\pm 12\%$), and subtle spatial translations ($\pm 6\%$).
   - Validation and test splits receive only deterministic ImageNet tensor normalization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).

---

## 3. Deep Learning Model Architectures

### A. Custom CNN (From Scratch)
Designed to provide a baseline without transfer learning priors:
- **Block 1:** Conv2d(3 $\rightarrow$ 32, $3\times3$) $\rightarrow$ BatchNorm2d $\rightarrow$ ReLU $\rightarrow$ Conv2d(32 $\rightarrow$ 32) $\rightarrow$ BatchNorm2d $\rightarrow$ ReLU $\rightarrow$ MaxPool2d($2\times2$) $\rightarrow$ Dropout2d(0.10)
- **Block 2:** Conv2d(32 $\rightarrow$ 64, $3\times3$) $\rightarrow$ BatchNorm2d $\rightarrow$ ReLU $\rightarrow$ Conv2d(64 $\rightarrow$ 64) $\rightarrow$ BatchNorm2d $\rightarrow$ ReLU $\rightarrow$ MaxPool2d($2\times2$) $\rightarrow$ Dropout2d(0.15)
- **Block 3:** Conv2d(64 $\rightarrow$ 128, $3\times3$) $\rightarrow$ BatchNorm2d $\rightarrow$ ReLU $\rightarrow$ Conv2d(128 $\rightarrow$ 128) $\rightarrow$ BatchNorm2d $\rightarrow$ ReLU $\rightarrow$ MaxPool2d($2\times2$) $\rightarrow$ Dropout2d(0.20)
- **Block 4:** Conv2d(128 $\rightarrow$ 256, $3\times3$) $\rightarrow$ BatchNorm2d $\rightarrow$ ReLU $\rightarrow$ AdaptiveAvgPool2d(($2\times2$)) $\rightarrow$ Dropout(0.25)
- **Classifier:** Flatten $\rightarrow$ Linear(1024 $\rightarrow$ 256) $\rightarrow$ BatchNorm1d $\rightarrow$ ReLU $\rightarrow$ Dropout(0.30) $\rightarrow$ Linear(256 $\rightarrow$ 6)

### B. ResNet-18 (Transfer Learning)
- Leverages ImageNet-pretrained residual representation.
- Replaces final fully-connected layer (`fc`) with `Sequential(Dropout(0.3), Linear(512, 6))`.
- Preserves high-level visual representations while tuning the final stage (`layer4`) and classifier for waste-specific features.

### C. DenseNet-121 (Transfer Learning)
- Leverages dense layer-to-layer connectivity for maximal feature reuse and gradient preservation.
- Replaces standard ImageNet classifier with `Sequential(Dropout(0.3), Linear(1024, 6))`.
- Fine-tunes `denseblock4`, `norm5`, and classifier head.

---

## 4. Empirical Model Evaluation & Benchmark Results

All three architectures were evaluated on 600 verified samples from the independent test dataset:

### Academic Model Comparison Table

| Model Architecture | Parameters | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score | Inference Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Custom CNN (From Scratch)** | ~1.2M | 33.17% | 36.55% | 33.17% | 28.05% | 28.05% | **18.52 ms/sample** |
| **ResNet-18 (Transfer Learning)** | ~11.2M | 71.00% | 74.10% | 71.00% | 71.63% | 71.63% | **17.95 ms/sample** ⚡ |
| **DenseNet-121 (Transfer Learning)** | ~7.0M | **72.67%** 🏆 | **74.43%** | **72.67%** | **72.97%** 🏆 | **72.97%** | **47.62 ms/sample** |

### Class-Wise Detailed Performance Matrix (Precision / Recall / F1)

| Waste Stream | Custom CNN (P / R / F1) | ResNet-18 (P / R / F1) | DenseNet-121 (P / R / F1) |
|:---|:---:|:---:|:---:|
| **Cardboard** | 43.0% / 74.0% / 54.4% | 86.6% / 84.0% / 85.3% | 80.9% / 89.0% / 84.8% |
| **Glass** | 28.2% / 64.0% / 39.1% | 84.9% / 73.0% / 78.5% | 85.4% / 76.0% / 80.4% |
| **Metal** | 26.4% / 28.0% / 27.2% | 80.5% / 70.0% / 74.9% | 77.8% / 70.0% / 73.7% |
| **Paper** | 50.0% / 3.0% / 5.7% | 63.9% / 53.0% / 57.9% | 72.7% / 56.0% / 63.3% |
| **Plastic** | 25.9% / 14.0% / 18.2% | 47.9% / 78.0% / 59.3% | 50.0% / 70.0% / 58.3% |
| **Trash** | 45.7% / 16.0% / 23.7% | 81.0% / 68.0% / 73.9% | 79.8% / 75.0% / 77.3% |

---

## 5. Two-Dustbin Segregation System

EcoSort AI strictly adheres to the municipal dual-stream segregation protocol:

| Predicted Category | Recommended Dustbin | Segregation Action |
|:---|:---:|:---|
| **Cardboard** | 🟦 **Blue – Dry Waste** | Flatten boxes; ensure dry & clean |
| **Glass** | 🟦 **Blue – Dry Waste** | Empty & rinse; wrap broken shards securely |
| **Metal** | 🟦 **Blue – Dry Waste** | Rinse food residue; recycle cans & caps |
| **Paper** | 🟦 **Blue – Dry Waste** | Keep unsoiled by food/oil; flatten |
| **Plastic** | 🟦 **Blue – Dry Waste** | Empty, rinse, and dry containers |
| **Trash** | ⚠️ **User Verification Required** | Inspect: Organic $\rightarrow$ 🟩 Green Bin; Dry Residual $\rightarrow$ 🟦 Blue Bin; Hazardous $\rightarrow$ Special Drop-off |

> **Crucial Segregation Rule:** The model does **not** possess a distinct wet/biological organic class. Therefore, **Trash** predictions never blindly default to the Green bin. Users are instructed to inspect the item:
> - **Biodegradable / Organic (food scraps, peels):** Place in **GREEN DUSTBIN**.
> - **Accepted Dry Residual (mixed non-recyclable wrappers):** Place in designated municipal residual container.
> - **Hazardous or Special Waste:** Follow local specialized collection rules.

---

## 6. Step-by-Step Disposal Instructions

- **Cardboard:**
  1. Remove unwanted packaging (plastic tape, styrofoam, bubble wrap).
  2. Ensure it is completely clean and dry.
  3. Flatten it to conserve space.
  4. Place it in the blue dry-waste bin.
- **Glass:**
  1. Empty any remaining liquid or contents.
  2. Check whether the container can be reused or returned.
  3. Handle broken glass carefully and wrap it securely in thick paper.
  4. Follow local municipal glass collection rules.
- **Metal:**
  1. Empty the container thoroughly.
  2. Remove food residue or labels.
  3. Rinse briefly with water if soiled.
  4. Place it in the blue bin if accepted locally.
- **Paper:**
  1. Check that the paper is clean and dry (free from grease or oil).
  2. Separate it from food waste or wet materials.
  3. Fold or flatten it neatly.
  4. Place it in the blue dry-waste bin.
- **Plastic:**
  1. Empty all liquid or food contents.
  2. Rinse if sticky or greasy.
  3. Allow it to dry thoroughly.
  4. Place it in the blue bin if accepted locally.
- **Trash:**
  1. Inspect the item carefully.
  2. Determine whether it is biodegradable organic waste or dry residual waste.
  3. Select the appropriate disposal route (Green bin for organics; local residual for dry non-recyclables).
  4. Follow local rules for hazardous, electronic, or sanitary waste.

---

## 7. Project Structure

```text
ECOSORT/
├── Dataset/                                # Original Parquet data files
│   ├── test-00000-of-00001.parquet
│   ├── train-00000-of-00003.parquet
│   ├── train-00001-of-00003.parquet
│   ├── train-00002-of-00003.parquet
│   └── validation-00000-of-00001.parquet
├── processed_dataset/                      # Extracted & structured dataset
│   ├── train/                              # 6 class subfolders
│   ├── val/                                # 6 class subfolders
│   ├── test/                               # 6 class subfolders
│   ├── dataset_summary.json                # Extraction statistics
│   ├── class_weights.json                  # Inverse frequency weights
│   └── dataset_report.md                   # Preprocessing markdown report
├── models/                                 # Checkpoints and evaluation
│   ├── custom_cnn_best.pth                 # Custom CNN PyTorch checkpoint
│   ├── resnet18_best.pth                   # ResNet-18 PyTorch checkpoint
│   ├── densenet121_best.pth                # DenseNet-121 PyTorch checkpoint
│   ├── evaluation_results.json             # Empirical test benchmark results
│   ├── custom_cnn_history.json             # Epoch loss/accuracy history
│   ├── resnet18_history.json               # Epoch loss/accuracy history
│   ├── densenet121_history.json            # Epoch loss/accuracy history
│   └── plots/                              # Generated evaluation graphs
│       ├── confusion_matrix_custom_cnn.png
│       ├── confusion_matrix_resnet18.png
│       ├── confusion_matrix_densenet121.png
│       ├── model_metrics_comparison.png
│       ├── training_loss_curves.png
│       └── training_acc_curves.png
├── src/                                    # Modular Python source code
│   ├── __init__.py
│   ├── dataset_loader.py                   # DataLoaders, augmentation & normalization
│   ├── preprocessing.py                    # Parquet extraction & validation
│   ├── custom_cnn.py                       # From-scratch CNN architecture
│   ├── resnet_model.py                     # ResNet-18 transfer learning model
│   ├── densenet_model.py                   # DenseNet-121 transfer learning model
│   ├── train.py                            # Training pipeline with AdamW & EarlyStopping
│   ├── evaluate.py                         # Test set evaluation & metric plots
│   ├── predict.py                          # Image inference engine & cache
│   └── recommendations.py                  # Two-dustbin segregation rule engine
├── app.py                                  # Professional Streamlit web application
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## 8. Installation & Usage Guide

### Step 1: Environment Setup
Ensure Python 3.10+ is installed. Clone the repository and install required packages:
```bash
pip install -r requirements.txt
```

### Step 2: Dataset Preprocessing
To extract, resize to 224×224 RGB, verify data integrity, and generate dataset summaries:
```bash
python src/preprocessing.py
```

### Step 3: Train All Models
Train Custom CNN, ResNet-18, and DenseNet-121 with AdamW optimizer, learning rate scheduling, and early stopping:
```bash
python src/train.py --model all --epochs 5 --batch_size 32
```

### Step 4: Evaluate Models
Run academic evaluation across the test dataset to compute Accuracy, Precision, Recall, F1-scores, Confusion Matrices, and Inference Latency:
```bash
python src/evaluate.py
```

### Step 5: Launch the Streamlit Application
Start the interactive EcoSort AI web application:
```bash
streamlit run app.py
```

---

## 9. Technology Stack

- **Deep Learning Framework:** PyTorch & Torchvision
- **Data Engineering:** PyArrow, Pandas, NumPy
- **Image Processing:** Pillow (PIL)
- **Evaluation & Metrics:** Scikit-Learn
- **Visualization:** Matplotlib, Plotly Graph Objects
- **Web Interface:** Streamlit (Custom Forest-Green Modern CSS)

---

## 10. Academic Attribution & License

- **Dataset Credit:** HuggingFace `TrashBusters/combined`
- **Application & Implementation:** Developed for academic deep learning research under the MIT License.
