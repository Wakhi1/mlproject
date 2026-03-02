import os
import logging
import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from app import models, dependencies
from app.routes.predict import _model, _encoders, _columns, load_model_artifacts

router = APIRouter(prefix="/predict", tags=["Bulk Prediction"])
logger = logging.getLogger(__name__)

# Ensure model is loaded (it will be when the module loads, but reload if needed)
# Optionally call load_model_artifacts() again if needed

@router.post("/bulk")
async def bulk_predict(
    file: UploadFile = File(...),
    current_user: models.User = Depends(dependencies.get_current_active_user)
):
    """
    Upload a CSV or Excel file with taxpayer data.
    Returns predictions as a CSV download.
    """
    # Validate file type
    if not (file.filename.endswith('.csv') or file.filename.endswith(('.xls', '.xlsx'))):
        raise HTTPException(400, "Only CSV or Excel files allowed")

    # Read file into pandas
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(400, f"Error reading file: {str(e)}")

    # Check if model artifacts are loaded
    if _model is None or _encoders is None or _columns is None:
        # Attempt to reload once
        load_model_artifacts()
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not available")

    # Check required columns (excluding target and tin)
    required = [col for col in _columns if col not in ['is_non_compliant', 'tin']]
    missing = set(required) - set(df.columns)
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}")

    # Preprocess
    df_input = df[required].copy()
    for col, le in _encoders.items():
        if col in df_input.columns:
            try:
                df_input[col] = le.transform(df_input[col].astype(str))
            except ValueError:
                df_input[col] = -1
    X = df_input[_columns]

    # Predict
    try:
        # Model may return probabilities; handle similarly to single prediction
        pred_proba = _model.predict(X)
        # If it returns 2D array, take positive class probability
        if isinstance(pred_proba, (pd.DataFrame, pd.Series, list)) and pred_proba.ndim > 1 and pred_proba.shape[1] > 1:
            probabilities = pred_proba[:, 1]
        else:
            probabilities = pred_proba

        predictions = (probabilities >= 0.5).astype(int)
    except Exception as e:
        logger.error(f"Bulk prediction failed: {e}")
        raise HTTPException(500, "Prediction error")

    # Add predictions to result
    result_df = df.copy()
    result_df['predicted_risk'] = predictions
    result_df['risk_probability'] = probabilities
    result_df['risk_level'] = result_df['predicted_risk'].map({0: 'Low', 1: 'High'})

    # Return as CSV
    stream = io.StringIO()
    result_df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    return response