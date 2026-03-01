from fastapi import FastAPI
from app.routes import auth, predict
from app.database import engine, Base

# Create database tables (optional, if you want SQLAlchemy to create them)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Compliance Risk API with Auth",
    description="Predict compliance risk. Register/Login to get JWT token.",
    version="2.0.0",
)

# Include routers
app.include_router(auth.router)
app.include_router(predict.router)

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "model_loaded": True}