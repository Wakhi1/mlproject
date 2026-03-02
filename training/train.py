import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Set MLflow tracking URI (from environment or default)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("compliance_risk")

# Load data
df = pd.read_csv('ers_sample_dataset.csv')
target = 'is_non_compliant'
X = df.drop(columns=[target, 'tin'])
y = df[target].astype(int)

# Encode categoricals (same as before)
cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

# Start MLflow run
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 5)

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc", auc)

    # Save encoders and columns as artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(encoders, "artifacts/encoders.pkl")
    joblib.dump(X.columns.tolist(), "artifacts/columns.pkl")
    mlflow.log_artifacts("artifacts", artifact_path="preprocessing")

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Register the model in MLflow Model Registry
    mlflow.register_model(f"runs:/{run.info.run_id}/model", "ComplianceRiskModel")

print(f"Run ID: {run.info.run_id}")