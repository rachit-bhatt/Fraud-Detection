from __future__ import annotations

import os
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature names and numeric values")
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class FraudPredictionService:
    """Load the registry champion and provide fraud predictions."""

    def __init__(self, model_uri: str | None = None) -> None:
        self.model_uri = model_uri or os.getenv(
            "MLFLOW_MODEL_URI",
            "models:/fraud-detection-model@champion",
        )
        self.model = None

    def load(self) -> None:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        self.model = mlflow.sklearn.load_model(self.model_uri)

    def predict(self, features: dict[str, float], threshold: float = 0.5) -> dict[str, Any]:
        if self.model is None:
            self.load()

        row = pd.DataFrame([features])
        expected_features = getattr(self.model, "feature_names_in_", None)
        if expected_features is not None:
            missing_features = set(expected_features) - set(row.columns)
            if missing_features:
                raise ValueError(f"Missing features: {sorted(missing_features)}")
            row = row.loc[:, expected_features]
        prediction = self.model.predict(row)
        result: dict[str, Any] = {"prediction": int(prediction[0])}

        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(row)[0][1])
            result["fraud_probability"] = probability
            result["threshold"] = threshold
            result["prediction"] = int(probability >= threshold)

        return result


service = FraudPredictionService()
app = FastAPI(title="Fraud Detection API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    try:
        service.load()
        return {"status": "ok", "model_uri": service.model_uri}
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        return service.predict(request.features, request.threshold)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
