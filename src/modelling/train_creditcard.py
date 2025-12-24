import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Utils
from src.modelling.utils import apply_smote, calculate_metrics, plot_confusion, stratified_cv

# -----------------------------
# 1️⃣ Load Feature-Engineered Credit Card Dataset
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
data_path = ROOT_DIR / "data" / "processed" / "creditcard_final.csv"
df = pd.read_csv(data_path)


# Target
target_col = "Class"
y = df[target_col]
X = df.drop(columns=[target_col])

# -----------------------------
# 2️⃣ Identify Categorical vs Numerical Features
# -----------------------------
# Usually, credit card dataset is all numerical (V1-V28, Amount, Time)
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# -----------------------------
# 3️⃣ Preprocessing Pipeline
# -----------------------------
# Credit card dataset is mostly numerical → just scale
preprocessor = Pipeline([
    ('scaler', StandardScaler())
])

X_scaled = pd.DataFrame(preprocessor.fit_transform(X), columns=X.columns)

# -----------------------------
# 4️⃣ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# 5️⃣ Apply SMOTE to Training Data
# -----------------------------
X_train_res, y_train_res = apply_smote(X_train, y_train)

# -----------------------------
# 6️⃣ Baseline Model → Logistic Regression
# -----------------------------
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_res, y_train_res)

# Predictions
y_pred_lr = lr_model.predict(X_test)
y_probs_lr = lr_model.predict_proba(X_test)[:,1]

# Metrics
metrics_lr = calculate_metrics(y_test, y_pred_lr, y_probs_lr)
print("=== Logistic Regression Metrics ===")
print(metrics_lr)

plot_confusion(y_test, y_pred_lr, title="LR Confusion Matrix")

# -----------------------------
# 7️⃣ Ensemble Models
# -----------------------------
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='logloss', random_state=42
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42
    )
}

ensemble_metrics = {}

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:,1]
    metrics = calculate_metrics(y_test, y_pred, y_probs)
    ensemble_metrics[name] = metrics
    print(f"=== {name} Metrics ===")
    print(metrics)
    plot_confusion(y_test, y_pred, title=f"{name} Confusion Matrix")

# -----------------------------
# 8️⃣ Cross-Validation (Optional)
# -----------------------------
print("=== Stratified K-Fold CV for RandomForest ===")
df_cv, summary_cv = stratified_cv(models["RandomForest"], X_scaled, y, cv=2)
print(df_cv)
print("CV Summary:")
print(summary_cv)

# -----------------------------
# 9️⃣ Save Best Model
# -----------------------------
# Assume RandomForest is best based on metrics
best_model = models["RandomForest"]
joblib.dump(best_model, "models/creditcard_best_model.pkl")
print("Best model saved as creditcard_best_model.pkl")
