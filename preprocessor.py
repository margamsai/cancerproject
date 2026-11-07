"""Data preprocessing module designed to prevent data leakage."""

from typing import Tuple, Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder


class Preprocessor:
    """Manages data cleaning, categorical encoding, scaling, and splitting without leakage."""

    SCALERS = {
        "StandardScaler (Z-Score)": StandardScaler,
        "MinMaxScaler (0 - 1)": MinMaxScaler,
        "RobustScaler (Outlier-resistant)": RobustScaler,
        "None (No Scaling)": None,
    }

    def __init__(self):
        self.scaler = None
        self.scaler_name = "None (No Scaling)"
        self.target_col = None
        self.target_encoder = None
        self.target_mapping = {}  # {original_label: encoded_int}
        self.feature_names = []
        self.feature_dtypes = {}
        self.imputation_values = {}  # {col: value} computed only on training set
        self.category_encoders = {}  # {col: LabelEncoder} for categorical features
        self.class_names = []  # e.g., ["Benign", "Malignant"]

    def clean_raw_data(
        self,
        df: pd.DataFrame,
        drop_duplicates: bool = True,
        cols_to_drop: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Perform initial row/column cleaning before splitting.

        Note: Dropping duplicates and dropping irrelevant identifiers (like 'id', 'patient_id')
        is performed prior to split as it does not leak target distributions.
        """
        df_clean = df.copy()

        # Drop user-selected useless columns (e.g., patient ID, unnamed indexing cols)
        if cols_to_drop:
            existing_drop = [c for c in cols_to_drop if c in df_clean.columns]
            if existing_drop:
                df_clean = df_clean.drop(columns=existing_drop)

        # Drop unnamed columns commonly found in CSV exports (e.g. Unnamed: 32)
        unnamed_cols = [c for c in df_clean.columns if "unnamed" in str(c).lower()]
        if unnamed_cols:
            df_clean = df_clean.drop(columns=unnamed_cols)

        if drop_duplicates:
            df_clean = df_clean.drop_duplicates()

        return df_clean

    def prepare_and_split(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: int = 42,
        num_impute_strategy: str = "median",
        cat_impute_strategy: str = "mode",
        scaling_strategy: str = "StandardScaler (Z-Score)",
        malignant_label_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform leakage-free preprocessing:
        1. Separate X and y.
        2. Encode target y.
        3. Split into train & test.
        4. Fit imputers & encoders ONLY on train, transform both train & test.
        5. Fit scaler ONLY on train, transform both train & test.
        """
        self.target_col = target_col
        self.scaler_name = scaling_strategy

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        # Drop rows where target is missing
        clean_df = df.dropna(subset=[target_col]).copy()

        # Separate X and y
        y_raw = clean_df[target_col]
        X_raw = clean_df.drop(columns=[target_col])

        # Encode target y
        y_encoded, class_names, target_map = self._encode_target(y_raw, malignant_label_hint)
        self.class_names = class_names
        self.target_mapping = target_map

        # Check stratification feasibility
        stratify_col = None
        if stratify:
            min_class_count = y_encoded.value_counts().min()
            if min_class_count >= 2:
                stratify_col = y_encoded
            else:
                stratify_col = None

        # Train / Test split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )

        # Handle features preprocessing strictly on train, then transform test
        X_train_proc, X_test_proc = self._fit_transform_features(
            X_train_raw,
            X_test_raw,
            num_impute_strategy,
            cat_impute_strategy,
            scaling_strategy
        )

        return {
            "X_train": X_train_proc,
            "X_test": X_test_proc,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "target_mapping": self.target_mapping,
            "preprocessor": self
        }

    def _encode_target(
        self,
        y_raw: pd.Series,
        malignant_label_hint: Optional[str] = None
    ) -> Tuple[pd.Series, List[str], Dict[Any, int]]:
        """Encode target into binary (0 and 1) or multiclass integers with oncology semantic alignment."""
        unique_vals = y_raw.unique()

        # If already integer binary (0, 1)
        if set(unique_vals).issubset({0, 1}):
            mapping = {0: 0, 1: 1}
            class_names = ["Negative/Benign (0)", "Positive/Malignant (1)"]
            return y_raw.astype(int), class_names, mapping

        # String or categorical handling
        unique_str = [str(v).strip() for v in unique_vals]

        # Check for cancer standard notations: M/B, Malignant/Benign, Cancer/Normal, Positive/Negative
        malignant_keywords = ["m", "malignant", "cancer", "positive", "1", "diseased", "tumorous", "high"]
        benign_keywords = ["b", "benign", "normal", "healthy", "negative", "0", "non-cancer", "low"]

        target_map = {}
        for val in unique_vals:
            val_str = str(val).strip().lower()
            if malignant_label_hint and str(val).strip().lower() == malignant_label_hint.lower():
                target_map[val] = 1
            elif any(val_str == kw for kw in malignant_keywords):
                target_map[val] = 1
            elif any(val_str == kw for kw in benign_keywords):
                target_map[val] = 0

        # If mapping is fully resolved for 2 classes
        if len(target_map) == len(unique_vals) and set(target_map.values()) == {0, 1}:
            inv_map = {v: k for k, v in target_map.items()}
            class_names = [f"Benign ({inv_map[0]})", f"Malignant ({inv_map[1]})"]
            y_enc = y_raw.map(target_map).astype(int)
            return y_enc, class_names, target_map

        # Fallback to standard scikit-learn LabelEncoder
        le = LabelEncoder()
        y_enc_arr = le.fit_transform(y_raw)
        self.target_encoder = le
        mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
        class_names = [f"Class {idx}: {cls}" for idx, cls in enumerate(le.classes_)]
        return pd.Series(y_enc_arr, index=y_raw.index, name=y_raw.name), class_names, mapping

    def _fit_transform_features(
        self,
        X_train_raw: pd.DataFrame,
        X_test_raw: pd.DataFrame,
        num_impute_strategy: str,
        cat_impute_strategy: str,
        scaling_strategy: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit imputers, encoders, and scaler on X_train_raw, then transform both X_train and X_test."""
        X_tr = X_train_raw.copy()
        X_te = X_test_raw.copy()

        num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_tr.select_dtypes(exclude=[np.number]).columns.tolist()

        # 1. Impute Numerical Features (compute statistic ONLY on training data)
        self.imputation_values = {}
        for col in num_cols:
            if num_impute_strategy == "mean":
                fill_val = X_tr[col].mean()
            elif num_impute_strategy == "median":
                fill_val = X_tr[col].median()
            elif num_impute_strategy == "mode":
                mode_vals = X_tr[col].mode()
                fill_val = mode_vals[0] if not mode_vals.empty else 0.0
            else:
                fill_val = 0.0

            if pd.isna(fill_val):
                fill_val = 0.0

            self.imputation_values[col] = float(fill_val)
            X_tr[col] = X_tr[col].fillna(fill_val)
            X_te[col] = X_te[col].fillna(fill_val)

        # 2. Impute and Encode Categorical Features
        self.category_encoders = {}
        for col in cat_cols:
            mode_vals = X_tr[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else "Missing"
            self.imputation_values[col] = fill_val

            X_tr[col] = X_tr[col].fillna(fill_val).astype(str)
            X_te[col] = X_te[col].fillna(fill_val).astype(str)

            # Fit label encoder on train
            le = LabelEncoder()
            le.fit(X_tr[col])
            self.category_encoders[col] = le

            # Transform train
            X_tr[col] = le.transform(X_tr[col])

            # Transform test with fallback for unseen categories
            known_classes = set(le.classes_)
            X_te[col] = X_te[col].apply(lambda v: v if v in known_classes else le.classes_[0])
            X_te[col] = le.transform(X_te[col])

        self.feature_names = X_tr.columns.tolist()

        # 3. Feature Scaling (fit ONLY on train, transform both)
        scaler_cls = self.SCALERS.get(scaling_strategy, None)
        if scaler_cls is not None:
            self.scaler = scaler_cls()
            X_tr_scaled = self.scaler.fit_transform(X_tr)
            X_te_scaled = self.scaler.transform(X_te)

            X_tr_proc = pd.DataFrame(X_tr_scaled, columns=self.feature_names, index=X_tr.index)
            X_te_proc = pd.DataFrame(X_te_scaled, columns=self.feature_names, index=X_te.index)
        else:
            self.scaler = None
            X_tr_proc = X_tr
            X_te_proc = X_te

        return X_tr_proc, X_te_proc

    def transform_new_samples(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Transform new patient record(s) for live inference using fitted preprocessors."""
        df_new = new_data.copy()

        # Ensure all required features are present
        for col in self.feature_names:
            if col not in df_new.columns:
                df_new[col] = self.imputation_values.get(col, 0.0)

        # Retain exact feature order
        df_new = df_new[self.feature_names].copy()

        # Apply imputations and encodings
        for col in self.feature_names:
            fill_val = self.imputation_values.get(col, 0.0)
            if col in self.category_encoders:
                df_new[col] = df_new[col].fillna(str(fill_val)).astype(str)
                le = self.category_encoders[col]
                known = set(le.classes_)
                df_new[col] = df_new[col].apply(lambda v: v if v in known else le.classes_[0])
                df_new[col] = le.transform(df_new[col])
            else:
                df_new[col] = pd.to_numeric(df_new[col], errors="coerce").fillna(fill_val)

        # Apply scaling
        if self.scaler is not None:
            scaled_arr = self.scaler.transform(df_new)
            df_new = pd.DataFrame(scaled_arr, columns=self.feature_names, index=df_new.index)

        return df_new
