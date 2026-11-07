"""Inference engine and model persistence utilities with comprehensive clinical reporting."""

from typing import Dict, Any, Tuple, Optional, List
import io
import time
import random
import datetime
import pandas as pd
import numpy as np
import joblib


class Predictor:
    """Manages manual single-patient inference, batch CSV inference, model serialization,
    and automated clinical diagnostic report generation.
    """

    @staticmethod
    def predict_single(
        model: Any,
        preprocessor: Any,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make a prediction for a single patient record."""
        df_input = pd.DataFrame([input_data])
        X_proc = preprocessor.transform_new_samples(df_input)

        raw_pred = model.predict(X_proc)[0]
 
        # Calculate prediction probabilities if supported
        probabilities = {}
        confidence = 0.0
        malignant_prob = None

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_proc)[0]
                for idx, p in enumerate(proba):
                    label = preprocessor.class_names[idx] if idx < len(preprocessor.class_names) else f"Class {idx}"
                    probabilities[label] = round(float(p) * 100, 2)

                confidence = round(float(np.max(proba)) * 100, 2)
                # If binary, determine malignancy probability
                if len(proba) == 2:
                    malignant_prob = round(float(proba[1]) * 100, 2)
            except Exception:
                pass

        # Resolve human-readable predicted label
        class_names = preprocessor.class_names
        predicted_label = class_names[raw_pred] if raw_pred < len(class_names) else f"Class {raw_pred}"

        # Risk classification
        if malignant_prob is not None:
            if malignant_prob < 35.0:
                risk_tier = "Low Risk (Likely Benign)"
                risk_color = "#10B981"
            elif malignant_prob < 65.0:
                risk_tier = "Moderate / Indeterminate Risk"
                risk_color = "#F59E0B"
            else:
                risk_tier = "High Risk (Likely Malignant)"
                risk_color = "#EF4444"
        else:
            risk_tier = "Classified"
            risk_color = "#2563EB"

        return {
            "raw_prediction": int(raw_pred),
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "malignant_probability": malignant_prob,
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "processed_input": X_proc
        }

    @staticmethod
    def generate_clinical_report(
        model: Any,
        preprocessor: Any,
        input_data: Dict[str, Any],
        raw_train_X: Optional[pd.DataFrame] = None,
        raw_train_y: Optional[pd.Series] = None,
        model_name: str = "Trained Classifier",
        patient_meta: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Generate a structured, printable clinical diagnostic pathology report."""
        pred_res = Predictor.predict_single(model, preprocessor, input_data)

        # Generate unique report metadata
        now = datetime.datetime.now()
        report_id = f"ONCO-RPT-{now.strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
        timestamp_str = now.strftime("%B %d, %Y - %H:%M:%S")

        patient_info = {
            "patient_id": (patient_meta or {}).get("patient_id") or f"PT-{random.randint(10000, 99999)}",
            "patient_name": (patient_meta or {}).get("patient_name") or "Anonymous Patient",
            "age": (patient_meta or {}).get("age") or "Not Specified",
            "gender": (patient_meta or {}).get("gender") or "Female",
            "referring_physician": (patient_meta or {}).get("physician") or "Dr. Oncology Clinical Staff",
            "specimen_type": "Fine Needle Aspirate (FNA) Nuclear Biopsy"
        }

        is_malignant = "malignant" in pred_res["predicted_label"].lower() or pred_res["raw_prediction"] == 1

        # Analyze individual biomarkers against baseline distributions
        biomarker_table = []
        abnormal_features = []

        if raw_train_X is not None and raw_train_y is not None:
            benign_mask = raw_train_y == 0
            malignant_mask = raw_train_y == 1

            for feat in preprocessor.feature_names:
                if feat in input_data and feat in raw_train_X.columns:
                    val = float(input_data[feat])
                    b_mean = float(raw_train_X.loc[benign_mask, feat].mean()) if benign_mask.any() else 0.0
                    b_std = float(raw_train_X.loc[benign_mask, feat].std()) if benign_mask.any() else 1.0
                    m_mean = float(raw_train_X.loc[malignant_mask, feat].mean()) if malignant_mask.any() else 0.0

                    # Check deviation from benign mean
                    z_from_benign = (val - b_mean) / b_std if b_std > 0 else 0.0
                    is_elevated = z_from_benign > 1.5

                    status_label = "Elevated / Atypical" if is_elevated else "Within Normal Range"

                    if is_elevated:
                        abnormal_features.append({
                            "feature": feat,
                            "value": round(val, 4),
                            "benign_baseline": round(b_mean, 4),
                            "deviation_z": round(z_from_benign, 2)
                        })

                    biomarker_table.append({
                        "feature": feat,
                        "patient_value": round(val, 4),
                        "benign_reference": round(b_mean, 4),
                        "malignant_reference": round(m_mean, 4),
                        "status": status_label,
                        "is_abnormal": is_elevated
                    })
        else:
            for feat, val in input_data.items():
                biomarker_table.append({
                    "feature": feat,
                    "patient_value": round(float(val), 4),
                    "benign_reference": "N/A",
                    "malignant_reference": "N/A",
                    "status": "Recorded",
                    "is_abnormal": False
                })

        # Sort abnormal features by degree of elevation
        abnormal_features.sort(key=lambda x: x.get("deviation_z", 0), reverse=True)

        # Automated Pathology Narrative Summary
        if is_malignant:
            narrative = (
                f"Computational analysis performed via {model_name} indicates an oncological finding consistent with "
                f"MALIGNANT CARCINOMA with a model confidence of {pred_res['confidence']}%. "
                f"Marked nuclear irregularities were observed across {len(abnormal_features)} cellular parameters. "
                f"Primary morphological drivers include elevated {', '.join([a['feature'] for a in abnormal_features[:3]]) or 'overall cellular dimensions'}. "
                f"Immediate clinical correlation, core needle histopathological biopsy, and oncological staging are strongly recommended."
            )
            recommendations = [
                "Schedule immediate urgent consultation with an oncology specialist.",
                "Recommend confirmatory ultrasound-guided Core Needle Biopsy (CNB) for definitive histopathology.",
                "Conduct Immunohistochemistry (IHC) profiling (ER, PR, HER2/neu, Ki-67).",
                "Perform bilateral diagnostic mammography and regional lymph node staging."
            ]
        else:
            narrative = (
                f"Computational evaluation via {model_name} demonstrates findings consistent with BENIGN LESION "
                f"with a model confidence of {pred_res['confidence']}%. Cellular and nuclear measurements fall within "
                f"the expected reference limits for non-malignant tissue. "
                f"Findings suggest regular follow-up with clinical observation and routine surveillance intervals."
            )
            recommendations = [
                "Continue standard annual or biennial screening mammography and clinical breast examination.",
                "Advise patient to self-monitor and report any palpable changes, discharge, or cutaneous abnormalities.",
                "Follow-up radiological imaging recommended in 6 to 12 months as indicated by treating physician."
            ]

        return {
            "report_id": report_id,
            "timestamp": timestamp_str,
            "patient_info": patient_info,
            "model_used": model_name,
            "diagnosis": "MALIGNANT (CARCINOMA INDICATED)" if is_malignant else "BENIGN (NON-MALIGNANT)",
            "is_malignant": is_malignant,
            "confidence": pred_res["confidence"],
            "probabilities": pred_res["probabilities"],
            "risk_tier": pred_res["risk_tier"],
            "risk_color": pred_res["risk_color"],
            "narrative": narrative,
            "recommendations": recommendations,
            "abnormal_features": abnormal_features[:5],
            "biomarkers": biomarker_table
        }

    @staticmethod
    def predict_batch(
        model: Any,
        preprocessor: Any,
        batch_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Run inference on a batch of patient records."""
        df_clean = batch_df.copy()
        unnamed = [c for c in df_clean.columns if "unnamed" in str(c).lower()]
        if unnamed:
            df_clean = df_clean.drop(columns=unnamed)

        if preprocessor.target_col and preprocessor.target_col in df_clean.columns:
            features_df = df_clean.drop(columns=[preprocessor.target_col])
        else:
            features_df = df_clean

        X_proc = preprocessor.transform_new_samples(features_df)
        raw_preds = model.predict(X_proc)

        class_names = preprocessor.class_names
        pred_labels = [
            class_names[p] if p < len(class_names) else f"Class {p}"
            for p in raw_preds
        ]

        results_df = batch_df.copy()
        results_df["Predicted_Class"] = pred_labels

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_proc)
                if proba.shape[1] == 2:
                    results_df["Confidence (%)"] = np.max(proba, axis=1).round(4) * 100
                    results_df["Malignant_Probability (%)"] = proba[:, 1].round(4) * 100
            except Exception:
                pass

        return results_df

    @staticmethod
    def export_pipeline(model: Any, preprocessor: Any, model_name: str) -> io.BytesIO:
        """Serialize trained model and preprocessor into a downloadable joblib binary buffer."""
        bundle = {
            "model": model,
            "preprocessor": preprocessor,
            "model_name": model_name,
            "feature_names": preprocessor.feature_names,
            "class_names": preprocessor.class_names,
            "target_mapping": preprocessor.target_mapping
        }
        buffer = io.BytesIO()
        joblib.dump(bundle, buffer)
        buffer.seek(0)
        return buffer
