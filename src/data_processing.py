# src/data_processing.py
"""
Module: data_processing
Purpose: Clean and preprocess raw fraud and credit card datasets.
This produces professionally processed CSVs that are ready for feature engineering.
Outputs:
- data/processed/fraud_data_processed.csv
- data/processed/creditcard_processed.csv
"""

import pandas as pd
from pathlib import Path

# -----------------------------
# Load Functions
# -----------------------------
def load_fraud_data(fraud_csv_path: str, ip_csv_path: str) -> pd.DataFrame:
    """Load fraud data and map IP addresses to countries"""
    try:
        fraud_df = pd.read_csv(fraud_csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Fraud data file not found: {fraud_csv_path}")
    except Exception as e:
        raise Exception(f"Error reading fraud CSV: {e}")

    try:
        ip_df = pd.read_csv(ip_csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"IP CSV file not found: {ip_csv_path}")
    except Exception as e:
        raise Exception(f"Error reading IP CSV: {e}")

    # Convert IP to numeric
    fraud_df['ip_address'] = pd.to_numeric(fraud_df['ip_address'], errors='coerce')
    ip_df['lower_bound_ip_address'] = pd.to_numeric(ip_df['lower_bound_ip_address'], errors='coerce')
    ip_df['upper_bound_ip_address'] = pd.to_numeric(ip_df['upper_bound_ip_address'], errors='coerce')

    # Map IP to country
    def map_ip_to_country(ip):
        row = ip_df[(ip_df['lower_bound_ip_address'] <= ip) & (ip <= ip_df['upper_bound_ip_address'])]
        if not row.empty:
            return row['country'].values[0]
        else:
            return "Unknown"

    fraud_df['country'] = fraud_df['ip_address'].apply(map_ip_to_country)

    return fraud_df

def load_creditcard_data(credit_csv_path: str) -> pd.DataFrame:
    """Load credit card dataset"""
    try:
        return pd.read_csv(credit_csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Credit card CSV not found: {credit_csv_path}")
    except Exception as e:
        raise Exception(f"Error reading credit card CSV: {e}")

# -----------------------------
# Clean Functions
# -----------------------------
def clean_fraud_data(fraud_df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning of fraud dataset"""
    try:
        # Convert timestamps
        fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'], errors='coerce')
        fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'], errors='coerce')

        # Drop rows with missing critical info
        fraud_df = fraud_df.dropna(subset=['signup_time', 'purchase_time', 'purchase_value', 'age'])

        # Fill missing age with median
        fraud_df['age'] = fraud_df['age'].fillna(fraud_df['age'].median())

        # Remove duplicates
        fraud_df = fraud_df.drop_duplicates()
    except Exception as e:
        raise Exception(f"Error cleaning fraud data: {e}")
    return fraud_df

def clean_creditcard_data(credit_df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning of credit card dataset"""
    try:
        credit_df = credit_df.drop_duplicates()
        numeric_cols = credit_df.columns.drop('Class')
        credit_df[numeric_cols] = credit_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        credit_df = credit_df.dropna()
    except Exception as e:
        raise Exception(f"Error cleaning credit card data: {e}")
    return credit_df

# -----------------------------
# Save Function
# -----------------------------
def save_processed_data(df: pd.DataFrame, filename: str, folder: str = "data/processed"):
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(folder) / filename, index=False)
        print(f"Saved: {folder}/{filename}")
    except Exception as e:
        raise Exception(f"Error saving {filename}: {e}")

# -----------------------------
# Main Processing
# -----------------------------
def process_all():
    try:
        # Fraud dataset
        fraud_df = load_fraud_data("data/raw/fraud_data.csv", "data/raw/ip.csv")
        fraud_clean = clean_fraud_data(fraud_df)
        save_processed_data(fraud_clean, "fraud_data_processed.csv")

        # Credit Card dataset
        credit_df = load_creditcard_data("data/raw/credit_card.csv")
        credit_clean = clean_creditcard_data(credit_df)
        save_processed_data(credit_clean, "creditcard_processed.csv")

    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    process_all()
