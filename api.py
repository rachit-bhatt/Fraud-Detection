"""FastAPI inference service for the MLflow registry champion."""
from __future__ import annotations
import os
from typing import Any
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from monitoring import PredictionMonitor

class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="All trained numeric feature names and values")
    threshold: float | None = Field(None, ge=0.0, le=1.0)

class FraudPredictionService:
    def __init__(self, model_uri: str | None = None) -> None:
        self.model_uri = model_uri or os.getenv("MLFLOW_MODEL_URI", "models:/fraud-detection-model@champion")
        self.model: Any | None = None
        self.monitor = PredictionMonitor()
    def load(self) -> None:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        self.model = mlflow.sklearn.load_model(self.model_uri)
    def predict(self, features: dict[str, float], threshold: float | None = None) -> dict[str, Any]:
        if self.model is None: self.load()
        row = pd.DataFrame([features]); expected = list(getattr(self.model, "feature_names_in_", []))
        missing, unexpected = set(expected) - set(row), set(row) - set(expected)
        if missing or unexpected:
            raise ValueError(f"Feature schema mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        row = row.loc[:, expected]
        score = float(self.model.predict_proba(row)[0][1]) if hasattr(self.model, "predict_proba") else None
        effective_threshold = 0.5 if threshold is None else threshold
        response = {"prediction": int(score >= effective_threshold) if score is not None else int(self.model.predict(row)[0]),
                    "fraud_probability": score, "threshold": effective_threshold, "model_uri": self.model_uri}
        self.monitor.log_prediction(row.iloc[0].to_dict(), response)
        return response

service = FraudPredictionService()
app = FastAPI(title="Fraud Detection API", version="1.0.0")
@app.get("/health")
def health() -> dict[str, str]:
    try:
        service.load(); return {"status": "ok", "model_uri": service.model_uri}
    except Exception as error: raise HTTPException(status_code=503, detail=str(error)) from error
@app.get("/metrics")
def metrics() -> dict[str, Any]: return service.monitor.summary()
@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    try: return service.predict(request.features, request.threshold)
    except Exception as error: raise HTTPException(status_code=400, detail=str(error)) from error
