import os
import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
import mlflow
import mlflow.pyfunc
from app import models, dependencies
from app.models import ComplianceInput, ComplianceOutput

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Set up logging
logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_NAME = "ComplianceRiskModel"

# Global variables for model and preprocessing artifacts
_model = None
_encoders = None
_columns = None

def load_model_artifacts():
    """Load the latest production model and preprocessing artifacts from MLflow."""
    global _model, _encoders, _columns
    try:
        client = mlflow.tracking.MlflowClient()
        # Try to get the latest model in Production stage
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if latest_versions:
            model_version = latest_versions[0].version
            model_uri = f"models:/{MODEL_NAME}/Production"
            logger.info(f"Loading model version {model_version} from Production stage")
        else:
            # Fallback to latest version (any stage)
            model_uri = f"models:/{MODEL_NAME}/latest"
            logger.info("No Production model found, loading latest version")

        _model = mlflow.pyfunc.load_model(model_uri)

        # Download preprocessing artifacts from the same run
        run_id = latest_versions[0].run_id if latest_versions else None
        if not run_id:
            # If no versions, try to get the latest run from experiment
            experiment = client.get_experiment_by_name("compliance_risk")
            if experiment:
                runs = client.search_runs(experiment.experiment_id, max_results=1)
                if runs:
                    run_id = runs[0].info.run_id
        if run_id:
            # Download encoders and columns from artifacts
            artifact_uri = client.download_artifacts(run_id, "preprocessing")
            import joblib
            _encoders = joblib.load(os.path.join(artifact_uri, "encoders.pkl"))
            _columns = joblib.load(os.path.join(artifact_uri, "columns.pkl"))
        else:
            logger.warning("Could not find run_id, falling back to local artifacts")
            _load_local_artifacts()
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        # Fallback to local artifacts
        _load_local_artifacts()

def _load_local_artifacts():
    """Fallback to local ml_artifacts folder."""
    global _encoders, _columns
    try:
        import joblib
        _encoders = joblib.load('ml_artifacts/encoders.pkl')
        _columns = joblib.load('ml_artifacts/columns.pkl')
        logger.info("Loaded local artifacts")
    except Exception as e:
        logger.error(f"Failed to load local artifacts: {e}")
        raise RuntimeError("Model artifacts not found. Run training first.")

# Load model at startup
load_model_artifacts()

@router.post("", response_model=ComplianceOutput)
async def predict(
    data: ComplianceInput,
    current_user: models.User = Depends(dependencies.get_current_active_user)
):
    """
    Predict compliance risk for a single taxpayer.
    """
    if _model is None or _encoders is None or _columns is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Apply label encoders to categorical columns
    for col, le in _encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = le.transform(input_df[col].astype(str))
            except ValueError:
                # Unseen category -> assign -1
                input_df[col] = -1

    # Ensure columns in training order
    input_df = input_df[_columns]

    # Predict
    try:
        # pyfunc model returns numpy array
        pred_proba = _model.predict(input_df)[0]  # For binary classification, this returns probability of positive class
        # For sklearn models loaded as pyfunc, it might return both classes; we need to check output shape
        # If it returns 2D array with probabilities for each class, take column 1.
        # To be safe, we'll assume it returns probability of positive class (common for mlflow.sklearn)
        if isinstance(pred_proba, (list, pd.Series, pd.DataFrame)) and len(pred_proba) > 1:
            prob = pred_proba[1]  # second column
        else:
            prob = pred_proba

        pred = 1 if prob >= 0.5 else 0
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

    return ComplianceOutput(
        prediction=int(pred),
        risk_probability=round(float(prob), 4),
        risk_level="High" if pred == 1 else "Low"
    )