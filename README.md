# 🎗️ Cancer Prediction Machine Learning System
### React & Node.js Frontend + Python Backend

An end-to-end Machine Learning web application for model training, algorithm comparison, and patient prediction report generation.

---

## ⚠️ Disclaimer

> **This application is an educational machine learning demonstration.**
> **It is NOT a medical device and must NOT be used for clinical diagnosis or treatment.**

---

## 🎯 Features

1. **Dataset Management & Exploration**:
   - Upload custom CSV or Excel datasets (or use the built-in Wisconsin Breast Cancer benchmark).
   - Automatic target column detection with manual selection support.
   - Summary statistics and class balance visualization.

2. **Preprocessing Pipeline**:
   - Zero data leakage design.
   - Missing value imputation (median, mean, mode).
   - Feature scaling (StandardScaler, MinMaxScaler, RobustScaler, or Raw).
   - Stratified train/test split.

3. **Multi-Model Training & Benchmark**:
   - Train and compare 6 machine learning algorithms:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - XGBoost
   - 5-Fold Cross Validation.
   - Leaderboard showing Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
   - One-click model pipeline export (`.joblib`).

4. **Prediction & Report Generation**:
   - Input patient measurements manually or use quick presets (Benign, Malignant, Average).
   - Generates a clean patient assessment report:
     - Predicted Classification (Benign / Malignant)
     - Model confidence percentage and risk level
     - Analysis summary narrative
     - Notable feature value comparisons
     - Recommendations
   - Print or save as PDF (`Ctrl + P`), or download as a text report.
   - Batch evaluation by uploading a CSV/Excel file.

---

## 📂 Project Structure

```
cancer project/
├── start.js                        # Node.js orchestrator (runs backend + frontend)
├── run_app.bat                     # 1-Click Windows batch launcher
├── package.json                    # Root orchestration scripts
├── requirements.txt                # Python backend dependencies
│
├── backend/
│   └── main.py                     # Python Flask API server (Port 8000)
│
├── frontend/                       # React 18 + Vite Frontend (Port 3000)
│   ├── package.json
│   ├── vite.config.js              # Proxy forwarding /api -> http://127.0.0.1:8000
│   ├── index.html
│   └── src/
│       ├── App.jsx                 # Main application dashboard
│       ├── api.js                  # Axios API client
│       ├── index.css               # Clean modern dark theme & print styling
│       └── main.jsx                # React mount point
│
├── src/                            # Python ML Engine
│   ├── data_loader.py              # File reader, validation, target detection
│   ├── preprocessor.py             # Data cleaning, scaling, stratified splitting
│   ├── model_trainer.py            # 6 Classifiers + K-Fold CV
│   ├── evaluator.py                # Performance metrics, confusion matrix, ROC-AUC
│   └── predictor.py                # Inference and report generator
│
└── data/
    └── sample_breast_cancer.csv    # Sample benchmark dataset
```

---

## 🚀 How to Run the Project

### Option 1: 1-Click Launcher (Windows)
Double-click:
📁 **`run_app.bat`**

### Option 2: From Terminal
```bash
npm start
```

Then open your browser to:
👉 **http://localhost:3000**

---

## 📖 Step-by-Step Guide

1. **Step 1 - Dataset**: Open the app. The Wisconsin Breast Cancer dataset is loaded automatically, or upload your own CSV/Excel file.
2. **Step 2 - Preprocess**: Review your preprocessing settings and click **"Run Preprocessing Pipeline"**.
3. **Step 3 - Train Models**: Go to the **Train Models** tab and click **"Train All Models"**. View the leaderboard and click any model to see its metrics and confusion matrix.
4. **Step 4 - Prediction & Report**: Go to the **Prediction & Report** tab. Click **"Benign Case"** or **"Malignant Case"** (or input values manually), then click **"Generate Prediction Report"**.
5. **Step 5 - Print / Export**: Click **"Print / PDF"** to save a PDF or print the report.
