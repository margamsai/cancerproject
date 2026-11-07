"""Model evaluation metrics, confusion matrix computation, and ROC-AUC calculation."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)


class Evaluator:
    """Computes comprehensive evaluation metrics and diagnostics for classification models."""

    @staticmethod
    def evaluate_model(
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate a trained model on test data."""
        y_pred = model.predict(X_test)

        # Check probability support
        y_prob = None
        has_proba = hasattr(model, "predict_proba")
        if has_proba:
            try:
                proba_arr = model.predict_proba(X_test)
                # For binary classification, take column 1
                if proba_arr.shape[1] == 2:
                    y_prob = proba_arr[:, 1]
                else:
                    y_prob = proba_arr
            except Exception:
                y_prob = None

        # Handle multiclass vs binary average
        is_binary = len(np.unique(y_test)) <= 2
        avg_mode = "binary" if is_binary else "weighted"

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average=avg_mode, zero_division=0))
        rec = float(recall_score(y_test, y_pred, average=avg_mode, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average=avg_mode, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        # ROC Curve & AUC
        fpr, tpr, roc_auc = None, None, None
        if is_binary and y_prob is not None:
            try:
                fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
                fpr = fpr_arr.tolist()
                tpr = tpr_arr.tolist()
                roc_auc = float(auc(fpr_arr, tpr_arr))
            except Exception:
                fpr, tpr, roc_auc = None, None, None

        # Detailed classification report
        clf_report = classification_report(
            y_test,
            y_pred,
            target_names=class_names if class_names and len(class_names) == len(np.unique(y_test)) else None,
            output_dict=True,
            zero_division=0
        )

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "fpr": fpr,
            "tpr": tpr,
            "classification_report": clf_report,
            "is_binary": is_binary
        }

    @staticmethod
    def get_feature_importances(model: Any, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """Extract feature importances or linear coefficients from models."""
        importances = None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            metric_label = "Importance"
        elif hasattr(model, "coef_"):
            coef = model.coef_
            # If 2D (e.g. binary logistic regression), take first row
            importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
            metric_label = "Absolute Coefficient"

        if importances is not None and len(importances) == len(feature_names):
            df_imp = pd.DataFrame({
                "Feature": feature_names,
                metric_label: importances
            }).sort_values(by=metric_label, ascending=False).reset_index(drop=True)
            return df_imp

        return None

    @staticmethod
    def compare_models(evaluations: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Create a comparative leaderboard table from multiple model evaluations."""
        records = []
        for model_name, eval_data in evaluations.items():
            if "error" in eval_data:
                continue
            records.append({
                "Model": model_name,
                "Accuracy (%)": round(eval_data.get("accuracy", 0.0) * 100, 2),
                "Precision (%)": round(eval_data.get("precision", 0.0) * 100, 2),
                "Recall (%)": round(eval_data.get("recall", 0.0) * 100, 2),
                "F1 Score (%)": round(eval_data.get("f1_score", 0.0) * 100, 2),
                "ROC-AUC": eval_data.get("roc_auc", "N/A"),
                "CV Accuracy (%)": (
                    round(eval_data.get("cv_mean_accuracy", 0.0) * 100, 2)
                    if eval_data.get("cv_mean_accuracy") is not None
                    else "N/A"
                ),
                "Training Time (s)": eval_data.get("training_time_sec", "N/A")
            })

        df_leaderboard = pd.DataFrame(records)
        if not df_leaderboard.empty and "Accuracy (%)" in df_leaderboard.columns:
            df_leaderboard = df_leaderboard.sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)
        return df_leaderboard
