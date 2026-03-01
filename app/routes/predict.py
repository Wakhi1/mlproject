from fastapi import APIRouter, Depends, HTTPException
import pandas as pd
import joblib
from app import models, dependencies
from app.models import ComplianceInput, ComplianceOutput  # we'll define these

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Load model artifacts (do this at module level, or use dependency)
try:
    model = joblib.load('ml_artifacts/model.pkl')
    encoders = joblib.load('ml_artifacts/encoders.pkl')
    columns = joblib.load('ml_artifacts/columns.pkl')
except FileNotFoundError:
    raise RuntimeError("Model artifacts not found. Run training script first.")

@router.post("", response_model=ComplianceOutput)
async def predict(
    data: ComplianceInput,
    current_user: models.User = Depends(dependencies.get_current_active_user)  # protected
):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Apply label encoders
    for col, le in encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = le.transform(input_df[col].astype(str))
            except ValueError:
                input_df[col] = -1  # unseen category

    input_df = input_df[columns]

    # Predict
    prob = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]

    return ComplianceOutput(
        prediction=int(pred),
        risk_probability=round(prob, 4),
        risk_level="High" if pred == 1 else "Low"
    )