"""Flask Backend for OncoAI Cancer Prediction Machine Learning System."""

import os
import sys
import io
from typing import Dict, Any, List, Optional

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model_trainer import ModelTrainer, XGBOOST_AVAILABLE
from src.evaluator import Evaluator
from src.predictor import Predictor

app = Flask(__name__)
# Enable CORS for React frontend (Vite port 3000 or 5173)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global in-memory workspace session state
workspace = {
    "df_raw": None,
    "dataset_name": "None",
    "target_col": None,
    "cols_to_drop": [],
    "split_data": None,
    "trained_models": {},
    "evaluations": {},
    "active_model_name": None
}


def load_benchmark_if_empty():
    """Ensure benchmark dataset is loaded by default."""
    if workspace["df_raw"] is None:
        sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_breast_cancer.csv")
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
        else:
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer(as_frame=True)
            df = data.frame.copy()
            df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
            df = df.drop(columns=["target"])
            df.columns = [c.replace(" ", "_") for c in df.columns]
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            df.to_csv(sample_path, index=False)

        workspace["df_raw"] = df
        workspace["dataset_name"] = "Wisconsin Breast Cancer Benchmark (569 rows, 30 features)"
        workspace["target_col"] = "diagnosis"
        workspace["cols_to_drop"] = []
        workspace["split_data"] = None
        workspace["trained_models"] = {}
        workspace["evaluations"] = {}


# Preload benchmark on startup
load_benchmark_if_empty()


# ==============================================================================
# WORKSPACE & DATASET ENDPOINTS
# ==============================================================================
@app.route("/api/status", methods=["GET"])
def get_status():
    """Return workspace pipeline status."""
    has_df = workspace["df_raw"] is not None
    has_split = workspace["split_data"] is not None

    return jsonify({
        "dataset_loaded": has_df,
        "dataset_name": workspace["dataset_name"],
        "target_col": workspace["target_col"],
        "rows": len(workspace["df_raw"]) if has_df else 0,
        "cols": workspace["df_raw"].shape[1] if has_df else 0,
        "preprocessed": has_split,
        "train_size": len(workspace["split_data"]["X_train"]) if has_split else 0,
        "test_size": len(workspace["split_data"]["X_test"]) if has_split else 0,
        "feature_count": len(workspace["split_data"]["feature_names"]) if has_split else 0,
        "trained_models": list(workspace["trained_models"].keys()),
        "active_model": workspace["active_model_name"],
        "xgboost_available": XGBOOST_AVAILABLE
    })


@app.route("/api/dataset/sample", methods=["POST"])
def load_sample():
    """Reset to Wisconsin benchmark dataset."""
    workspace["df_raw"] = None
    load_benchmark_if_empty()
    return get_dataset_summary()


@app.route("/api/dataset/upload", methods=["POST"])
def upload_dataset():
    """Upload custom CSV or Excel dataset."""
    if "file" not in request.files:
        return jsonify({"detail": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        contents = file.read()
        file_obj = io.BytesIO(contents)
        file_obj.name = file.filename

        df = DataLoader.load_file(file_obj)
        workspace["df_raw"] = df
        workspace["dataset_name"] = file.filename
        workspace["target_col"] = DataLoader.detect_target_column(df)
        workspace["cols_to_drop"] = []
        workspace["split_data"] = None
        workspace["trained_models"] = {}
        workspace["evaluations"] = {}

        return get_dataset_summary()
    except Exception as e:
        return jsonify({"detail": str(e)}), 400


@app.route("/api/dataset/summary", methods=["GET"])
def get_dataset_summary():
    """Return detailed dataset profiling and preview."""
    load_benchmark_if_empty()
    df = workspace["df_raw"]
    profile = DataLoader.inspect_dataset(df)

    preview = df.head(10).replace({np.nan: None}).to_dict(orient="records")
    col_specs = profile["missing_table"].to_dict(orient="records")

    return jsonify({
        "name": workspace["dataset_name"],
        "rows": profile["num_rows"],
        "cols": profile["num_cols"],
        "total_missing": profile["total_missing"],
        "duplicate_count": profile["duplicate_count"],
        "target_col": workspace["target_col"],
        "all_columns": df.columns.tolist(),
        "numeric_cols": profile["numeric_cols"],
        "preview": preview,
        "col_specs": col_specs,
        "cols_to_drop": workspace["cols_to_drop"]
    })


@app.route("/api/dataset/configure", methods=["POST"])
def configure_target():
    """Update target column and excluded columns."""
    data = request.get_json() or {}
    target_col = data.get("target_col")
    cols_to_drop = data.get("cols_to_drop", [])

    if workspace["df_raw"] is None:
        return jsonify({"detail": "No dataset loaded"}), 400
    if target_col not in workspace["df_raw"].columns:
        return jsonify({"detail": "Target column not in dataset"}), 400

    workspace["target_col"] = target_col
    workspace["cols_to_drop"] = cols_to_drop
    workspace["split_data"] = None
    workspace["trained_models"] = {}
    workspace["evaluations"] = {}
    return jsonify({"message": "Configuration updated successfully", "target_col": target_col})


@app.route("/api/dataset/eda", methods=["GET"])
def get_eda_data():
    """Return target distribution and correlation matrix data."""
    load_benchmark_if_empty()
    df = workspace["df_raw"]
    target_col = workspace["target_col"]

    target_dist = []
    if target_col and target_col in df.columns:
        val_counts = df[target_col].value_counts()
        total = float(len(df))
        for cls_val, count in val_counts.items():
            target_dist.append({
                "class": str(cls_val),
                "count": int(count),
                "percentage": round((count / total) * 100, 2)
            })

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col][:12]
    corr_data = {}
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr().round(3).fillna(0)
        corr_data = {
            "columns": num_cols,
            "values": corr_matrix.values.tolist()
        }

    return jsonify({
        "target_distribution": target_dist,
        "correlation": corr_data,
        "numeric_features": num_cols
    })


# ==============================================================================
# PREPROCESSING ENDPOINTS
# ==============================================================================
@app.route("/api/preprocess", methods=["POST"])
def run_preprocess():
    """Execute leakage-free preprocessing pipeline."""
    load_benchmark_if_empty()
    df = workspace["df_raw"]
    target_col = workspace["target_col"]
    data = request.get_json() or {}

    scaling_strategy = data.get("scaling_strategy", "StandardScaler (Z-Score)")
    num_impute_strategy = data.get("num_impute_strategy", "median")
    cat_impute_strategy = data.get("cat_impute_strategy", "mode")
    test_size = float(data.get("test_size", 0.2))
    stratify = bool(data.get("stratify", True))
    random_state = int(data.get("random_state", 42))
    drop_duplicates = bool(data.get("drop_duplicates", True))

    try:
        prep = Preprocessor()
        df_clean = prep.clean_raw_data(
            df,
            drop_duplicates=drop_duplicates,
            cols_to_drop=workspace["cols_to_drop"]
        )
        split_res = prep.prepare_and_split(
            df_clean,
            target_col=target_col,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
            num_impute_strategy=num_impute_strategy,
            cat_impute_strategy=cat_impute_strategy,
            scaling_strategy=scaling_strategy
        )

        workspace["split_data"] = split_res
        workspace["trained_models"] = {}
        workspace["evaluations"] = {}

        return jsonify({
            "message": "Preprocessing completed successfully",
            "train_samples": len(split_res["X_train"]),
            "test_samples": len(split_res["X_test"]),
            "features": split_res["feature_names"],
            "class_names": split_res["class_names"],
            "target_mapping": {str(k): int(v) for k, v in split_res["target_mapping"].items()}
        })
    except Exception as e:
        return jsonify({"detail": str(e)}), 400


# ==============================================================================
# MODEL TRAINING & EVALUATION ENDPOINTS
# ==============================================================================
def ensure_preprocessed():
    """Auto-preprocess with recommended defaults if not yet preprocessed."""
    if workspace["split_data"] is None:
        load_benchmark_if_empty()
        prep = Preprocessor()
        df_clean = prep.clean_raw_data(
            workspace["df_raw"],
            drop_duplicates=True,
            cols_to_drop=workspace["cols_to_drop"]
        )
        split_res = prep.prepare_and_split(
            df_clean,
            target_col=workspace["target_col"],
            test_size=0.2,
            stratify=True,
            random_state=42
        )
        workspace["split_data"] = split_res


@app.route("/api/train", methods=["POST"])
def train_models():
    """Train single or all models with Cross-Validation."""
    ensure_preprocessed()
    sd = workspace["split_data"]
    X_train, y_train = sd["X_train"], sd["y_train"]

    data = request.get_json() or {}
    model_name = data.get("model_name", "All")
    perform_cv = bool(data.get("perform_cv", True))
    cv_folds = int(data.get("cv_folds", 5))

    models_to_run = (
        [m for m in ModelTrainer.SUPPORTED_MODELS if m != "XGBoost" or XGBOOST_AVAILABLE]
        if model_name == "All"
        else [model_name]
    )

    trained_dict = {}
    eval_dict = {}

    for m_name in models_to_run:
        try:
            res = ModelTrainer.train_model(
                m_name,
                X_train,
                y_train,
                perform_cv=perform_cv,
                cv_folds=cv_folds
            )
            trained_dict[m_name] = res

            eval_res = Evaluator.evaluate_model(
                res["model"],
                sd["X_test"],
                sd["y_test"],
                class_names=sd["class_names"]
            )
            eval_res["training_time_sec"] = res["training_time_sec"]
            eval_res["cv_mean_accuracy"] = res["cv_mean_accuracy"]
            eval_dict[m_name] = eval_res
        except Exception as e:
            eval_dict[m_name] = {"error": str(e)}

    workspace["trained_models"].update(trained_dict)
    workspace["evaluations"].update(eval_dict)
    if models_to_run and models_to_run[0] in trained_dict:
        workspace["active_model_name"] = models_to_run[0]

    return get_evaluation_results()


@app.route("/api/models", methods=["GET"])
def get_evaluation_results():
    """Return model performance leaderboard and detailed diagnostics."""
    evals = workspace["evaluations"]
    if not evals:
        return jsonify({"leaderboard": [], "models": {}, "active_model": None})

    sd = workspace["split_data"]
    leaderboard_df = Evaluator.compare_models(evals)
    leaderboard_records = leaderboard_df.to_dict(orient="records")

    detailed_models = {}
    for m_name, ev in evals.items():
        if "error" in ev or m_name not in workspace["trained_models"]:
            continue

        model_obj = workspace["trained_models"][m_name]["model"]
        cm = ev["confusion_matrix"].tolist() if isinstance(ev["confusion_matrix"], np.ndarray) else ev["confusion_matrix"]

        imp_df = Evaluator.get_feature_importances(model_obj, sd["feature_names"])
        importances = imp_df.head(10).to_dict(orient="records") if imp_df is not None else []

        detailed_models[m_name] = {
            "accuracy": ev["accuracy"],
            "precision": ev["precision"],
            "recall": ev["recall"],
            "f1_score": ev["f1_score"],
            "roc_auc": ev["roc_auc"],
            "cv_accuracy": ev.get("cv_mean_accuracy"),
            "training_time": ev.get("training_time_sec"),
            "confusion_matrix": cm,
            "fpr": ev["fpr"],
            "tpr": ev["tpr"],
            "class_names": sd["class_names"],
            "feature_importances": importances
        }

    return jsonify({
        "leaderboard": leaderboard_records,
        "models": detailed_models,
        "active_model": workspace["active_model_name"]
    })


# ==============================================================================
# PREDICTION (INFERENCE) ENDPOINTS
# ==============================================================================
@app.route("/api/predict/features", methods=["GET"])
def get_predict_features():
    """Return feature names and stats (mean, min, max, std) for UI input generation."""
    ensure_preprocessed()
    sd = workspace["split_data"]
    raw_X = sd["X_train"]

    feature_info = []
    for feat in sd["feature_names"]:
        f_mean = float(raw_X[feat].mean()) if feat in raw_X.columns else 0.0
        f_std = float(raw_X[feat].std()) if feat in raw_X.columns and raw_X[feat].std() > 0 else 1.0
        f_min = float(raw_X[feat].min()) if feat in raw_X.columns else f_mean - 2 * f_std
        f_max = float(raw_X[feat].max()) if feat in raw_X.columns else f_mean + 2 * f_std

        feature_info.append({
            "name": feat,
            "mean": round(f_mean, 4),
            "std": round(f_std, 4),
            "min": round(f_min, 4),
            "max": round(f_max, 4),
            "benign_preset": round(max(f_min, f_mean - 0.75 * f_std), 4),
            "malignant_preset": round(min(f_max, f_mean + 1.25 * f_std), 4)
        })

    return jsonify({
        "features": feature_info,
        "class_names": sd["class_names"],
        "trained_models": list(workspace["trained_models"].keys())
    })


@app.route("/api/predict/single", methods=["POST"])
def predict_single():
    """Predict cancer classification and generate full clinical diagnostic report."""
    ensure_preprocessed()
    data = request.get_json() or {}
    model_name = data.get("model_name") or workspace["active_model_name"]

    if not model_name or model_name not in workspace["trained_models"]:
        train_models_internal("Random Forest")
        model_name = "Random Forest"

    model_obj = workspace["trained_models"][model_name]["model"]
    sd = workspace["split_data"]
    prep = sd["preprocessor"]
    features = data.get("features", {})
    patient_meta = data.get("patient_meta", {})

    report = Predictor.generate_clinical_report(
        model=model_obj,
        preprocessor=prep,
        input_data=features,
        raw_train_X=sd.get("X_train"),
        raw_train_y=sd.get("y_train"),
        model_name=model_name,
        patient_meta=patient_meta
    )
    return jsonify(report)


def train_models_internal(model_name: str = "Random Forest"):
    """Internal helper to train a default model."""
    ensure_preprocessed()
    sd = workspace["split_data"]
    res = ModelTrainer.train_model(model_name, sd["X_train"], sd["y_train"], perform_cv=False)
    workspace["trained_models"][model_name] = res
    workspace["active_model_name"] = model_name


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """Predict batch cases from uploaded CSV/Excel file."""
    ensure_preprocessed()
    if "file" not in request.files:
        return jsonify({"detail": "No file uploaded"}), 400

    file = request.files["file"]
    m_name = request.form.get("model_name") or workspace["active_model_name"] or "Random Forest"
    if m_name not in workspace["trained_models"]:
        train_models_internal(m_name)

    model_obj = workspace["trained_models"][m_name]["model"]
    prep = workspace["split_data"]["preprocessor"]

    contents = file.read()
    file_obj = io.BytesIO(contents)
    file_obj.name = file.filename

    batch_df = DataLoader.load_file(file_obj)
    out_df = Predictor.predict_batch(model_obj, prep, batch_df)

    preview = out_df.head(20).replace({np.nan: None}).to_dict(orient="records")
    return jsonify({
        "total_records": len(out_df),
        "preview": preview,
        "columns": out_df.columns.tolist()
    })


@app.route("/api/model/export", methods=["GET"])
def export_model():
    """Download serialized trained model pipeline (.joblib)."""
    m_name = request.args.get("model_name") or workspace["active_model_name"]
    if not m_name or m_name not in workspace["trained_models"]:
        return jsonify({"detail": "Model has not been trained"}), 400

    model_obj = workspace["trained_models"][m_name]["model"]
    prep = workspace["split_data"]["preprocessor"]
    buf = Predictor.export_pipeline(model_obj, prep, m_name)

    return send_file(
        buf,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"{m_name.lower().replace(' ', '_')}_pipeline.joblib"
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
