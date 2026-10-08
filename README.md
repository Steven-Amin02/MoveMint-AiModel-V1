# MoveMint AI Workstream: Edge-AI Transit Telemetry Pipeline

**MoveMint** is a Green-Tech mobile platform designed for Greater Cairo that combines carbon-optimized multimodal routing, passive commute tracking, and gamification. 

This repository contains the **Edge-AI Machine Learning subsystem** (responsible for verifying transportation modes via smartphone IMU sensor telemetry).

---

## 1. Project Architecture & Directory Layout

```
AI-2/
├── Data/
│   └── 28943915/              # Benchmark dataset (PLOS ONE, 2025 / Figshare)
│       ├── HTC/               # 10,000 pre-segmented windows (450 timesteps × 12 channels)
│       │   ├── htc_x_small.npy
│       │   └── htc_y_small.npy
│       └── SHL/               # ~5.4M continuous sensor recordings at 100 Hz
│           ├── Motion.txt
│           └── Label.txt
├── src/
│   ├── config.py              # Central hyperparameters, channels, and class definitions
│   ├── data/
│   │   ├── dataset.py         # PyTorch Dataset, stratified split, and DataLoader factory
│   │   ├── augmentations.py   # SO(3) 3D rotation, jitter, scaling, and time masking
│   │   └── shl_preprocessor.py# Decimator & windowing pipeline for raw SHL data
│   ├── models/
│   │   ├── baseline_cnn.py    # Lightweight 1D-CNN (<40K params) for mobile TFLite
│   │   └── baseline_features.py# 14-feature extractor from Fang et al. (Sensors 2016)
│   ├── utils/
│   │   ├── metrics.py         # Multi-class & MoveMint Anti-Cheat Security metrics
│   │   └── visualization.py   # Confusion matrices & loss/accuracy curve plotting
│   └── train.py               # End-to-end training, validation, and checkpointing
├── scripts/
│   ├── train_baseline_rf.py   # Classical ML baseline (Random Forest on 14 features)
│   └── export_onnx.py         # ONNX cross-platform export utility
├── tests/
│   └── test_pipeline.py       # Unit test suite verifying tensor flow & metrics
├── checkpoints/               # Persisted model weights (best_model.pt)
└── outputs/                   # Visualizations, confusion matrices, and exported models
```

---

## 2. Sensor Channel Specification (12 Channels)

All sensor time-series are sampled/decimated to **50 Hz** ($20\text{ ms}$ interval), matching Android's `SENSOR_DELAY_GAME`:

| Channels | Sensor Modality | Unit | Physical Significance |
| :---: | :--- | :--- | :--- |
| `0..2` | Triaxial Accelerometer ($X, Y, Z$) | Normalized ($m/s^2$) | Raw vehicle/body acceleration + static gravity |
| `3..5` | Triaxial Linear Accelerometer ($X, Y, Z$) | Normalized ($m/s^2$) | Pure dynamic motion (gravity removed) |
| `6..8` | Triaxial Gyroscope ($X, Y, Z$) | Normalized ($rad/s$) | Rotational turning rates & curve kinetics |
| `9..11` | Triaxial Magnetometer ($X, Y, Z$) | Normalized ($\mu T$) | Ambient geomagnetic field & track environment |

---

## 3. Transportation Mode Classes

```python
0: "Still"       # Stationary / baseline
1: "Walk"        # Micro-mobility
2: "Run"         # Micro-mobility
3: "Bike"        # Micro-mobility
4: "Car"         # Private vehicle (Negative Class / Anti-Cheat Target)
5: "Bus"         # Public transit (Positive Class / Reward Leg)
6: "Train"       # Public transit (Positive Class / Reward Leg)
7: "Subway"      # Public transit (Positive Class / Reward Leg)
```

---

## 4. MoveMint Anti-Cheat Operational Metrics

Unlike generic ML benchmarks that only report overall accuracy, MoveMint monitors critical operational risk metrics:

* **Car-to-Transit False Positive Rate ($FPR_{\text{car} \rightarrow \text{transit}}$):**
  $$\text{FPR} = \frac{\text{Car instances predicted as Bus, Train, or Subway}}{\text{Total actual Car instances}}$$
  * **Target:** $\le 5\%$. Ensures private car drivers cannot spoof public transit rewards.
* **Transit Verification Recall:**
  $$\text{Recall}_{\text{transit}} = \frac{\text{Actual Bus/Train/Subway instances correctly verified}}{\text{Total actual Bus/Train/Subway instances}}$$
  * **Target:** $\ge 90\%$. Ensures commuters are fairly awarded for sustainable travel.

---

## 5. Model Benchmarking Matrix

Evaluated on the held-out test partition of **1,500 unseen sensor windows** from the HTC dataset:

| Architecture | Paradigm | Test Accuracy | Macro F1 | Anti-Cheat Car FPR | Public Transit Recall | Model Size (TFLite) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1D-CNN** | Raw temporal Convolutions | 70.13% | 70.76% | 33.07% | 79.39% | 165 KB |
| **Multi-Scale ResNet** | Parallel Receptive Fields | 72.60% | 73.21% | 29.97% | 77.75% | 1.0 MB |
| **Random Forest (14 Feats)** | Handcrafted FFT & Jerk stats | 85.53% | 84.82% | 4.65% | 80.80% | N/A (Sklearn) |
| **Wavelet Multi-Scale S-CNN (W-SCNN)** | **Learnable DWT + Multi-Scale SE** | **80.13%** | **80.04%** | **20.16%** | **85.01%** | **165 KB (FP32) / 327 KB (INT8)** |

---

## 6. Production Model Architecture: Wavelet Multi-Scale S-CNN (`MoveMintWaveletSCNN`)

* **Type:** Discrete Wavelet Transform (DWT) Filterbank + Multi-Scale ResNet + Squeeze-and-Excitation + Dual Pooling (GAP + GMP).
* **Trainable Parameters:** $291{,}368$
* **Frequency Decomposition:** Haar/Biorthogonal depthwise 1D filterbank (`stride=2`) disentangles low-frequency macro kinetics (trend) from high-frequency road & motor vibration (detail).
* **Edge Inference Latency:** $< 3\text{ ms}$ on standard mobile ARM CPU.
* **TFLite Export:**
  * Float32: [`outputs/tflite/movemint_edge_model_float32.tflite`](file:///d:/Graduation-project-26/AI-2/outputs/tflite/movemint_edge_model_float32.tflite) (165 KB)
  * INT8: [`outputs/tflite/movemint_edge_model_int8.tflite`](file:///d:/Graduation-project-26/AI-2/outputs/tflite/movemint_edge_model_int8.tflite) (327 KB)
  * Native Android App Asset: [`android_test_app/app/src/main/assets/movemint_edge_model_float32.tflite`](file:///d:/Graduation-project-26/AI-2/android_test_app/app/src/main/assets/movemint_edge_model_float32.tflite)
