"""Data loading, validation, and automated exploration utilities."""

from typing import Tuple, Dict, Any, Optional
import io
import pandas as pd
import numpy as np


class DataLoader:
    """Handles dataset loading, validation, and initial profiling."""

    LIKELY_TARGET_NAMES = [
        "diagnosis", "target", "class", "cancer", "outcome", "label",
        "result", "malignancy", "status", "stage", "is_cancer", "tumor_type"
    ]

    @staticmethod
    def load_file(file_obj) -> pd.DataFrame:
        """Load uploaded file (CSV or Excel) into a pandas DataFrame.

        Parameters
        ----------
        file_obj : UploadedFile or file-like object
            Streamlit uploaded file object or path string.

        Returns
        -------
        pd.DataFrame
            Loaded dataframe.
        """
        filename = getattr(file_obj, "name", str(file_obj)).lower()

        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(file_obj)
            except UnicodeDecodeError:
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                df = pd.read_csv(file_obj, encoding="latin1")
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_obj)
        else:
            raise ValueError("Unsupported file format. Please upload a .csv, .xlsx, or .xls file.")

        DataLoader.validate_dataframe(df)
        return df

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> None:
        """Perform sanity checks on the loaded dataframe."""
        if df is None or df.empty:
            raise ValueError("Uploaded file is empty or could not be parsed.")
        if df.shape[1] < 2:
            raise ValueError("Dataset must contain at least 2 columns (features and target).")

    @staticmethod
    def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive exploratory profile of the dataset."""
        num_rows, num_cols = df.shape
        missing_series = df.isnull().sum()
        missing_pct = (missing_series / num_rows) * 100
        duplicate_count = int(df.duplicated().sum())

        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Count": missing_series.values,
            "Missing (%)": missing_pct.round(2).values,
            "Data Type": [str(t) for t in df.dtypes.values]
        }).sort_values(by="Missing Count", ascending=False)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        return {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "duplicate_count": duplicate_count,
            "total_missing": int(missing_series.sum()),
            "missing_table": missing_df,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "numeric_summary": df.describe().T if len(numeric_cols) > 0 else pd.DataFrame(),
            "categorical_summary": df.describe(include=["object", "category"]).T if len(categorical_cols) > 0 else pd.DataFrame(),
        }

    @classmethod
    def detect_target_column(cls, df: pd.DataFrame) -> Optional[str]:
        """Automatically identify the most probable target column."""
        cols_lower = {col.lower().strip(): col for col in df.columns}

        # Check explicit common name matches
        for target_hint in cls.LIKELY_TARGET_NAMES:
            for lower_name, original_col in cols_lower.items():
                if target_hint == lower_name or target_hint in lower_name:
                    return original_col

        # Look for binary or low cardinality columns (2-5 unique values) near edges
        candidate_cols = []
        for col in df.columns:
            n_unique = df[col].nunique(dropna=True)
            if 2 <= n_unique <= 5:
                candidate_cols.append(col)

        if candidate_cols:
            # Check last column first (canonical ML format)
            if df.columns[-1] in candidate_cols:
                return df.columns[-1]
            # Check first column
            if df.columns[0] in candidate_cols:
                return df.columns[0]
            return candidate_cols[0]

        # Default fallback: last column
        return df.columns[-1]
