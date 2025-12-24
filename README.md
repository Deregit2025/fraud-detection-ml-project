

```markdown
# Fraud Detection Project

**Author:** Dereje Derib  
**Challenge:** 10 Academy – Fraud Detection  
**Date:** December 2025  

---

## 🧠 Business Problem

Fraudulent transactions cause financial loss and damage customer trust.  
This project builds Machine Learning models to detect fraud in:

- 🛒 E-commerce transactions
- 💳 Credit card transactions

Key objectives:

- Detect fraudulent activities early  
- Minimize false positives to avoid disturbing legitimate users  
- Handle extreme class imbalance effectively  
- Build interpretable and production-ready models  

---

## 📂 Data Sources

| Dataset | Description |
|--------|------------|
| fraud_data.csv | E-commerce transaction dataset |
| credit_card.csv | Bank credit card transactions (anonymized PCA features) |
| ip_data.csv | IP → Country mapping |

All **raw datasets** are stored in:  
```

data/raw

```

---

## 🚀 Project Pipeline

### ✅ Task-1 — Data Preparation & Feature Engineering
1️⃣ **Data Processing (`src/data_processing.py`)**
- Handle missing values  
- Remove duplicates  
- Correct inconsistent formats  
- Save cleaned datasets → `data/processed`

2️⃣ **Feature Engineering (`src/feature_engineering.py`)**
- Time-based features (hour, weekday)
- Behavioural & frequency features
- Combined geolocation insights
- Final engineered datasets saved as:
```

fraud_data_final.csv
creditcard_final.csv

```

---

## ✅ Task-2 — Model Training & Evaluation

Two independent modeling pipelines are implemented.

---

### 🛒 Ecommerce Fraud Modeling
Script:
```

src/modelling/train_ecommerce.py

```

Models Trained:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Outputs:
- Precision, Recall, F1-Score, AUC-PR
- Best model saved:
```

models/ecommerce_best_model.pkl

```

---

### 💳 Credit Card Fraud Modeling
Script:
```

src/modelling/train_creditcard.py

```

Models Trained:
- Logistic Regression  
- Random Forest  
- XGBoost  
- LightGBM  
- Stratified K-Fold Cross-Validation (RandomForest)

Outputs:
- Model comparison metrics  
- Cross-validation results  
- Best model saved:
```

models/creditcard_best_model.pkl

```

---

## 📊 Evaluation Strategy

Because the datasets are highly imbalanced, we prioritize:

- **Recall** → catch as many frauds as possible  
- **Precision** → reduce false alarms  
- **F1-Score** → balanced performance  
- **AUC-PR (primary metric)** → best suited for imbalanced datasets  

---

## 🗂 Repository Structure

```

fraud_detection/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   ├── eda-fraud.ipynb
│   ├── eda-creditcard.ipynb
│   └── modeling.ipynb
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── utils/
│   │   └── balancing_utils.py
│   └── modelling/
│       ├── train_ecommerce.py
│       └── train_creditcard.py
├── requirements.txt
└── README.md

```

---

## ▶️ How to Run

### 1️⃣ Create & Activate Virtual Environment
```

python -m venv .venv
..venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/Mac

```

### 2️⃣ Install Dependencies
```

pip install -r requirements.txt

```

### 3️⃣ Run Data Processing
```

python -m src.data_processing

```

### 4️⃣ Run Feature Engineering
```

python -m src.feature_engineering

```

### 5️⃣ Train Ecommerce Models
```

python -m src.modelling.train_ecommerce

```

### 6️⃣ Train Credit Card Models
```

python -m src.modelling.train_creditcard

```

---

## 📌 Notes

- Raw datasets are not pushed to GitHub  
- Modular design enables easy extension  
- SMOTE utilities exist and will be explored further  
- Logs and saved models ensure reproducibility  

---

## 📍 Project Status

✔️ Task-1 Completed — Cleaning + Feature Engineering  
✔️ Task-2 Completed — Modeling Pipelines  
⬜ Task-3 — Explainability (SHAP)  
⬜ Deployment  
⬜ Final Reporting  

---

## 📜 License
Educational use under 10 Academy Program.



