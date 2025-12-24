import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

# -----------------------------
# 1️⃣ SMOTE Helper
# -----------------------------
def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to training data only.
    
    Args:
        X_train (pd.DataFrame or np.ndarray): Training features
        y_train (pd.Series or np.ndarray): Training target
        random_state (int): random seed
    
    Returns:
        X_res, y_res: balanced training features and target
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

# -----------------------------
# 2️⃣ Metrics Helper
# -----------------------------
def calculate_metrics(y_true, y_pred, y_probs=None):
    """
    Compute main evaluation metrics for imbalanced classification.

    Args:
        y_true: true labels
        y_pred: predicted labels
        y_probs: predicted probabilities (for AUC-PR)
    
    Returns:
        dict of metrics
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }
    
    if y_probs is not None:
        auc_pr = average_precision_score(y_true, y_probs)
        metrics["AUC-PR"] = auc_pr
    
    return metrics

# -----------------------------
# 3️⃣ Confusion Matrix Plot
# -----------------------------
def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot a confusion matrix using seaborn heatmap.

    Args:
        y_true: true labels
        y_pred: predicted labels
        title: plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

# -----------------------------
# 4️⃣ Stratified K-Fold CV Helper
# -----------------------------
def stratified_cv(model, X, y, cv=5, random_state=42):
    """
    Perform Stratified K-Fold CV and return metrics per fold.

    Args:
        model: sklearn-like estimator
        X: features
        y: target
        cv: number of folds
    
    Returns:
        pd.DataFrame of metrics per fold + mean/std
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    metrics_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Apply SMOTE on training fold
        X_res, y_res = apply_smote(X_train, y_train, random_state=random_state)

        # Fit model
        model.fit(X_res, y_res)

        # Predict
        y_pred = model.predict(X_val)
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_val)[:,1]
        else:
            y_probs = None

        # Compute metrics
        metrics = calculate_metrics(y_val, y_pred, y_probs)
        metrics["Fold"] = fold
        metrics_list.append(metrics)

    # Convert to DataFrame
    df_metrics = pd.DataFrame(metrics_list)
    summary = df_metrics.drop(columns="Fold").agg(["mean", "std"])
    
    return df_metrics, summary
