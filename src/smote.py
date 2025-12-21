# src/smote.py
"""
Module: smote
Purpose: Handle class imbalance for fraud and credit card datasets using SMOTE.
Outputs balanced datasets ready for model training.
"""

import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# -----------------------------
# Load final dataset
# -----------------------------
def load_final_data(filename: str, folder: str = "data/processed") -> pd.DataFrame:
    try:
        df = pd.read_csv(Path(folder) / filename)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Final dataset not found: {filename}")
    except Exception as e:
        raise Exception(f"Error loading final dataset {filename}: {e}")

# -----------------------------
# Separate features and target
# -----------------------------
def split_features_target(df: pd.DataFrame, target_col: str):
    try:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y
    except Exception as e:
        raise Exception(f"Error splitting features and target: {e}")

# -----------------------------
# Apply SMOTE
# -----------------------------
def apply_smote(X, y, random_state=42):
    try:
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)
        print(f"Before SMOTE: {y.value_counts().to_dict()}")
        print(f"After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
        return X_res, y_res
    except Exception as e:
        raise Exception(f"Error applying SMOTE: {e}")

# -----------------------------
# Split train-test
# -----------------------------
def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise Exception(f"Error in train-test split: {e}")

# -----------------------------
# Main function
# -----------------------------
def process_smote_for_all():
    try:
        # Fraud data
        fraud_df = load_final_data("fraud_data_final.csv")
        X_fraud, y_fraud = split_features_target(fraud_df, 'class')
        X_fraud_res, y_fraud_res = apply_smote(X_fraud, y_fraud)
        X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = stratified_train_test_split(X_fraud_res, y_fraud_res)

        # Credit card data
        credit_df = load_final_data("creditcard_final.csv")
        X_credit, y_credit = split_features_target(credit_df, 'Class')
        X_credit_res, y_credit_res = apply_smote(X_credit, y_credit)
        X_credit_train, X_credit_test, y_credit_train, y_credit_test = stratified_train_test_split(X_credit_res, y_credit_res)

        return {
            "fraud": (X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test),
            "credit": (X_credit_train, X_credit_test, y_credit_train, y_credit_test)
        }

    except Exception as e:
        print(f"[ERROR] SMOTE processing failed: {e}")
        return None

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    datasets = process_smote_for_all()
    if datasets:
        print("SMOTE processing completed successfully.")
