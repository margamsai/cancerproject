"""Machine Learning model training, tuning, and cross-validation module."""

from typing import Dict, Any, Tuple, Optional, List
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Optional XGBoost import with safe handling
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class ModelTrainer:
    """Manages training, cross-validation, and hyperparameter configuration for all supported models."""

    SUPPORTED_MODELS = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine (SVM)",
        "K-Nearest Neighbours (KNN)",
        "XGBoost",
    ]

    @staticmethod
    def get_model_instance(model_name: str, random_state: int = 42, **kwargs) -> Any:
        """Instantiate a model with sensible clinical prediction defaults."""
        if model_name == "Logistic Regression":
            c_val = kwargs.get("C", 1.0)
            max_iter = kwargs.get("max_iter", 1000)
            return LogisticRegression(C=c_val, max_iter=max_iter, random_state=random_state)

        elif model_name == "Decision Tree":
            max_depth = kwargs.get("max_depth", 6)
            criterion = kwargs.get("criterion", "gini")
            return DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)

        elif model_name == "Random Forest":
            n_estimators = kwargs.get("n_estimators", 100)
            max_depth = kwargs.get("max_depth", 8)
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )

        elif model_name == "Support Vector Machine (SVM)":
            c_val = kwargs.get("C", 1.0)
            kernel = kwargs.get("kernel", "rbf")
            return SVC(C=c_val, kernel=kernel, probability=True, random_state=random_state)

        elif model_name == "K-Nearest Neighbours (KNN)":
            n_neighbors = kwargs.get("n_neighbors", 5)
            weights = kwargs.get("weights", "uniform")
            return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, n_jobs=-1)

        elif model_name == "XGBoost":
            if not XGBOOST_AVAILABLE:
                raise ImportError(
                    "XGBoost is not installed or supported in the current environment. "
                    "Install with 'pip install xgboost' or choose another classifier."
                )
            n_estimators = kwargs.get("n_estimators", 100)
            learning_rate = kwargs.get("learning_rate", 0.1)
            max_depth = kwargs.get("max_depth", 5)
            return XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                eval_metric="logloss",
                random_state=random_state,
                use_label_encoder=False
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    @classmethod
    def train_model(
        cls,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int = 42,
        perform_cv: bool = False,
        cv_folds: int = 5,
        **hyperparams
    ) -> Dict[str, Any]:
        """Train a single model and calculate training latency and cross-validation scores."""
        model = cls.get_model_instance(model_name, random_state=random_state, **hyperparams)

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = round(time.time() - start_time, 4)

        cv_mean, cv_std = None, None
        if perform_cv:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
            cv_mean = round(float(np.mean(cv_scores)), 4)
            cv_std = round(float(np.std(cv_scores)), 4)

        return {
            "model_name": model_name,
            "model": model,
            "training_time_sec": training_time,
            "cv_mean_accuracy": cv_mean,
            "cv_std_accuracy": cv_std,
            "features": X_train.columns.tolist()
        }

    @classmethod
    def train_all_models(
        cls,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int = 42,
        perform_cv: bool = True,
        cv_folds: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """Train all supported models to enable comprehensive multi-algorithm comparison."""
        results = {}
        for name in cls.SUPPORTED_MODELS:
            if name == "XGBoost" and not XGBOOST_AVAILABLE:
                continue
            try:
                res = cls.train_model(
                    name,
                    X_train,
                    y_train,
                    random_state=random_state,
                    perform_cv=perform_cv,
                    cv_folds=cv_folds
                )
                results[name] = res
            except Exception as e:
                results[name] = {"error": str(e), "model": None}
        return results
