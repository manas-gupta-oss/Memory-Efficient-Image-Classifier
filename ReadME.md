# Memory-Efficient Image Classifier (C + OpenCV + Classical ML)

## Overview

This project implements a **low-memory, low-latency image classification pipeline** designed for edge devices without GPU acceleration.

The system replaces deep learning with a **classical computer vision + linear ML pipeline**, achieving competitive performance under strict resource constraints.

---

## Motivation

Modern CNN-based models (e.g., MobileNet) are effective but often:

* require high memory
* depend on hardware acceleration
* introduce unpredictable latency

This project explores a deterministic alternative:

* fixed feature extraction (HOG)
* dimensionality reduction (PCA)
* lightweight classifier (SVM / Logistic Regression)

---

## System Pipeline

```
Input Image (64×64 grayscale)
        ↓
Preprocessing (resize, normalize)
        ↓
HOG Feature Extraction (~3780 dims)
        ↓
PCA Compression (→ 64 dims)
        ↓
Linear Classifier (SVM)
        ↓
Prediction
```

---

## Key Features

* Memory-efficient (< 20 MB RAM target)
* Fast inference (< 50 ms per image target)
* No deep learning dependency in deployment
* Cross-language pipeline:

  * Python → training
  * C/C++ → inference

---

## Tech Stack

* C/C++ (inference engine)
* Python (training pipeline)
* OpenCV (feature extraction)
* NumPy / scikit-learn (ML components)

---

## Project Structure

```
.
├── python/
│   ├── train.py
│   ├── pca.py
│   └── export_model.py
│
├── c/
│   ├── main.cpp
│   ├── hog.cpp
│   ├── pca.cpp
│   └── classifier.cpp
│
├── model/
│   ├── weights.bin
│   ├── pca_matrix.bin
│   └── mean.bin
│
├── data/
│   └── dataset/
│
└── README.md
```

---

## Mathematical Formulation

### Gradient Computation

Gx = I(x+1,y) - I(x-1,y)
Gy = I(x,y+1) - I(x,y-1)

### HOG Descriptor

* Orientation histograms computed per cell
* Normalized over blocks

### PCA Projection

x_reduced = Wᵀ (x - μ)

### Classifier

y = wᵀ x + b

---

## Performance Targets

| Metric       | Target    |
| ------------ | --------- |
| RAM Usage    | < 20 MB   |
| Latency      | < 50 ms   |
| Feature Size | 3780 → 64 |

---

## Results (Example)

| Model          | Accuracy | Latency | RAM   |
| -------------- | -------- | ------- | ----- |
| HOG + SVM      | 92%      | 30 ms   | 12 MB |
| CNN (baseline) | 95%      | 120 ms  | 80 MB |

---

## Build Instructions (C++)

```bash
g++ -O3 -march=native main.cpp -o classifier `pkg-config --cflags --libs opencv4`
./classifier input.jpg
```

---

## Training Pipeline (Python)

```bash
python train.py
python export_model.py
```

---

## Optimization Techniques

* PCA-based dimensionality reduction
* Static memory allocation in C
* Feature quantization (optional int8)
* Cache-friendly data layout

---

## Limitations

* Lower accuracy compared to deep CNNs
* Sensitive to image scale and orientation
* Requires careful feature engineering

---

## Future Work

* SIMD optimization (AVX/NEON)
* Replace HOG with learned descriptors
* Hybrid model (tiny CNN + classical features)
* Hardware deployment (Raspberry Pi / MCU)

---

## Contributors

* C Implementation: Jagrav Singh
* Python Pipeline: Manas Gupta

---

## License

MIT License
