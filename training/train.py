"""
Model training script for Compliance Risk.
Loads CSV, encodes categoricals, trains a RandomForest, and saves artifacts.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load data
df = pd.read_csv('ers_sample_dataset.csv')

# Separate features and target
target = 'is_non_compliant'
X = df.drop(columns=[target, 'tin'])
y = df[target].astype(int)

# Identify categorical columns (works with both pandas <2.0 and >=2.0)
cat_cols = X.select_dtypes(include=['string', 'object']).columns.tolist()

# Encode categoricals
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Train model
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
model.fit(X, y)

# Save artifacts
os.makedirs('../ml_artifacts', exist_ok=True)
joblib.dump(model, '../ml_artifacts/model.pkl')
joblib.dump(encoders, '../ml_artifacts/encoders.pkl')
joblib.dump(X.columns.tolist(), '../ml_artifacts/columns.pkl')

print("✅ Model training complete. Files saved in ml_artifacts/")