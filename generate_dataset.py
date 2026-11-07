"""Utility script to generate the Wisconsin Breast Cancer benchmark dataset."""

import os
from sklearn.datasets import load_breast_cancer
import pandas as pd

def main():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    
    # Map target: In sklearn, 0 is Malignant, 1 is Benign
    df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
    df = df.drop(columns=["target"])
    
    # Clean up column names (standard formatting)
    df.columns = [col.replace(" ", "_") for col in df.columns]
    
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "sample_breast_cancer.csv")
    
    df.to_csv(output_file, index=False)
    print(f"Successfully exported benchmark dataset to {output_file} with shape {df.shape}")

if __name__ == "__main__":
    main()
