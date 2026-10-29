# KisanMitra AI — Benchmark & Evaluation Results

This report documents all experimental results, validation metrics, and performance benchmarks for **KisanMitra AI**. These metrics can be directly referenced in project reports and resume bullet points.

---

## 1. Executive Summary of Resume Metrics

| Metric | Target | Final Achieved | Status |
| :--- | :--- | :--- | :--- |
| **Disease Classification Accuracy** | $\ge 95.0\%$ | *Pending Phase 3* | Scheduled |
| **Disease Macro F1-Score** | $\ge 0.94$ | *Pending Phase 3* | Scheduled |
| **Mobile Model Latency (CPU)** | $< 100\text{ ms}$ | *Pending Phase 3* | Scheduled |
| **Mandi Price Forecast MAPE** | $\le 12.0\%$ | *Pending Phase 5* | Scheduled |
| **Mandi Price Forecast RMSE** | Baseline lowest | *Pending Phase 5* | Scheduled |
| **Yield Impact Calibration** | ICAR aligned | *Pending Phase 6* | Scheduled |

---

## 2. Crop Disease Classification Benchmarks (PlantVillage - 13 Classes)

### 2.1 Scope & Target Classes
- **Potato (3 classes):**
  1. Potato Early Blight (*Alternaria solani*)
  2. Potato Late Blight (*Phytophthora infestans*)
  3. Potato Healthy
- **Tomato (10 classes):**
  1. Tomato Bacterial Spot (*Xanthomonas*)
  2. Tomato Early Blight (*Alternaria solani*)
  3. Tomato Late Blight (*Phytophthora infestans*)
  4. Tomato Leaf Mold (*Passalora fulva*)
  5. Tomato Septoria Leaf Spot (*Septoria lycopersici*)
  6. Tomato Spider Mites (*Tetranychus urticae*)
  7. Tomato Target Spot (*Corynespora cassiicola*)
  8. Tomato Yellow Leaf Curl Virus (TYLCV)
  9. Tomato Mosaic Virus (ToMV)
  10. Tomato Healthy

### 2.2 Model Comparison (Transfer Learning)

| Model Architecture | Parameters | Train Accuracy | Val Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Inference Time (CPU) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MobileNetV3-Small (Baseline) | ~2.5M | - | - | - | - | - | - |
| MobileNetV3-Large | ~5.4M | - | - | - | - | - | - |
| EfficientNet-B0 | ~5.3M | - | - | - | - | - | - |

*(Note: Exact metrics will be populated upon completing training runs in Phase 3.)*

### 2.3 Confusion Matrix & Per-Class Breakdown
- Detailed per-class precision, recall, and false positive rates will be logged here.
- Key confusion boundary to monitor: Tomato Early Blight vs. Septoria Leaf Spot vs. Target Spot.

---

## 3. Mandi Price & Yield Forecasting Benchmarks

### 3.1 Data Splits & Protocol
- **Evaluation Rule:** Strict **time-based train/validation/test split** (never random shuffle) to prevent temporal data leakage.
- **Data Sources:** Agmarknet / data.gov.in daily mandi modal prices + Open-Meteo agro-meteorological metrics (temperature, precipitation, relative humidity).

### 3.2 Price Prediction Model Comparison

| Model Architecture | Horizon | Train Split Period | Test Split Period | RMSE (₹/Quintal) | MAE (₹/Quintal) | MAPE (%) | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Naïve Lag-7 Baseline | 7 days | - | - | - | - | - | - |
| Facebook Prophet | 7 days | - | - | - | - | - | - |
| XGBoost Regressor | 7 days | - | - | - | - | - | - |
| LSTM / GRU | 7 days | - | - | - | - | - | - |

*(Note: Exact metrics will be populated upon completing training runs in Phase 5.)*

---

## 4. Hardware & Environmental Details (For Resume Accuracy)

- **Training Environment:** Google Colab / Kaggle (NVIDIA T4 / P100 GPU) & Apple Silicon Local
- **Inference Hardware:** CPU / Apple MPS / Cloud Container (1 vCPU, 2GB RAM standard free-tier)
- **Frameworks:** PyTorch 2.x, Torchvision, Scikit-Learn, XGBoost, Streamlit, MLflow
