from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
from app.routes import auth, predict, bulk, model
from app.database import engine, Base
import logging

# Custom Prometheus metrics
login_attempts = Counter('login_attempts_total', 'Total login attempts', ['status'])
registrations = Counter('registrations_total', 'Total user registrations')
prediction_count = Counter('prediction_count', 'Predictions by result', ['result'])  # positive/negative
model_auc = Gauge('model_auc', 'Current model AUC')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
model_precision = Gauge('model_precision', 'Current model precision')
model_recall = Gauge('model_recall', 'Current model recall')
drift_score = Gauge('drift_score', 'Data drift score (0-1)')

# Logger
logger = logging.getLogger(__name__)

# Create database tables (if not exist)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Compliance Risk API with Auth & Model Management",
    description="Predict compliance risk. Register/Login to get JWT token. Bulk upload and model retraining available.",
    version="3.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation (includes default metrics)
instrumentator = Instrumentator().instrument(app)

# Expose metrics endpoint with custom metrics
@app.get("/metrics")
async def get_metrics():
    """Expose Prometheus metrics (including custom)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Startup event to expose metrics (optional)
@app.on_event("startup")
async def startup():
    # The instrumentator already exposes at /metrics by default if we call .expose(app)
    # But we want to include our custom metrics, so we use a custom endpoint above.
    # To avoid duplicate /metrics, we don't call instrumentator.expose(app)
    logger.info("Starting up, custom metrics enabled at /metrics")
    # You could also schedule periodic tasks to update drift_score, etc.

# Include routers
app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(bulk.router)
app.include_router(model.router)

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "model_loaded": True}