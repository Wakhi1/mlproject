from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from app import models, database, ml_utils, dependencies
import pandas as pd
from datetime import datetime
from typing import List

router = APIRouter(prefix="/model", tags=["Model Management"])

@router.get("/versions", response_model=List[dict])
def list_versions(db: Session = Depends(database.get_db)):
    """List all trained model versions with metrics."""
    versions = db.query(models.ModelVersion).order_by(models.ModelVersion.version.desc()).all()
    return [
        {
            "version": v.version,
            "created_at": v.created_at,
            "metrics": v.get_metrics(),
            "notes": v.notes
        }
        for v in versions
    ]

@router.post("/retrain")
async def retrain_model(
    file: UploadFile = File(...),
    notes: str = "",
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(database.get_db)
):
    """
    Upload a labeled CSV file (must include 'is_non_compliant').
    Retrains the model, updates artifacts, and records a new version.
    """
    # Validate file
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Only CSV files allowed")

    # Read CSV
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(400, f"Error reading file: {str(e)}")

    # Check required columns
    if 'is_non_compliant' not in df.columns:
        raise HTTPException(400, "CSV must contain 'is_non_compliant' column")

    # Prepare data: separate features, encode
    X, y, new_encoders, new_columns = ml_utils.prepare_training_data(df)

    # Train and evaluate
    model, metrics = ml_utils.train_and_evaluate(X, y)

    # Save new artifacts (overwrites previous)
    ml_utils.save_artifacts(model, new_encoders, new_columns)

    # Determine next version number
    last_version = db.query(models.ModelVersion).order_by(models.ModelVersion.version.desc()).first()
    next_version = (last_version.version + 1) if last_version else 1

    # Record version in database
    version_record = models.ModelVersion(
        version=next_version,
        notes=notes or f"Retrained on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    version_record.set_metrics(metrics)
    db.add(version_record)
    db.commit()
    db.refresh(version_record)

    return {
        "message": "Model retrained successfully",
        "version": next_version,
        "metrics": metrics
    }