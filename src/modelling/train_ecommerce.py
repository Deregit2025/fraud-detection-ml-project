import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path

# Utils
from src.modelling.utils import apply_smote, calculate_metrics, plot_confusion, stratified_cv

# load data
ROOT_DIR = Path(__file__).resolve().parents[2]
data_path = ROOT_DIR / "data" / "processed" / "fraud_data_final.csv"

df = pd.read_csv(data_path)





# Target
target_col = "class"

cols_to_drop = [target_col]

for col in ["user_id", "device_id"]:
    if col in df.columns:
        cols_to_drop.append(col)

X = df.drop(columns=cols_to_drop)
y = df[target_col]


# -----------------------------
# 2️⃣ Identify Categorical vs Numerical Features
# -----------------------------
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# -----------------------------
# 3️⃣ Preprocessing Pipeline
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# -----------------------------
# 4️⃣ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply preprocessing
X_train = pd.DataFrame(preprocessor.fit_transform(X_train))
X_test = pd.DataFrame(preprocessor.transform(X_test))

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
# 8️⃣ Cross-Validation (Optional but Recommended)
# -----------------------------
print("=== Stratified K-Fold CV for RandomForest ===")
df_cv, summary_cv = stratified_cv(models["RandomForest"], pd.DataFrame(X), y, cv=5)
print(df_cv)
print("CV Summary:")
print(summary_cv)

# -----------------------------
# 9️⃣ Save Best Model
# -----------------------------
# Here we assume RandomForest is the best based on metrics
best_model = models["RandomForest"]
joblib.dump(best_model, "models/ecommerce_best_model.pkl")
print("Best model saved as ecommerce_best_model.pkl")
