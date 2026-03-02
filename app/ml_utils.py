import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

ARTIFACTS_DIR = "ml_artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, "encoders.pkl")
COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, "columns.pkl")

def load_artifacts():
    """Load model, encoders, and feature columns."""
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, encoders, columns

def save_artifacts(model, encoders, columns):
    """Save model artifacts."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(columns, COLUMNS_PATH)

def preprocess_input(df, encoders, columns):
    """
    Apply label encoders to a DataFrame and ensure correct column order.
    Returns DataFrame ready for prediction.
    """
    # Apply encoders
    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError:
                # unseen category -> assign -1
                df[col] = -1
    # Ensure columns in training order
    return df[columns]

def prepare_training_data(df, target_col='is_non_compliant'):
    """
    Separate features and target, encode categoricals using new encoders.
    Returns X, y, and the fitted encoders and feature columns.
    """
    X = df.drop(columns=[target_col, 'tin'] if 'tin' in df.columns else [target_col])
    y = df[target_col].astype(int)

    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()

    # Encode categoricals
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    return X, y, encoders, X.columns.tolist()

def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    """
    Train a RandomForest and return model + metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
        "test_size": len(y_test)
    }
    return model, metrics