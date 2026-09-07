"""FastAPI inference service for an MLflow champion and its model contract."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field

from monitoring import PredictionMonitor


class PredictionRequest(BaseModel):
    """The threshold is intentionally absent: it belongs to the approved model."""

    features: dict[str, float] = Field(..., description="All contract feature names and numeric values")


class FraudPredictionService:
    def __init__(self, model_uri: str | None = None) -> None:
        self.model_uri = model_uri or os.getenv("MLFLOW_MODEL_URI", "models:/fraud-detection-model@champion")
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///runtime/mlflow.db")
        self.model: Any | None = None
        self.contract: dict[str, Any] | None = None
        self.model_version: str | None = None
        self.monitor = PredictionMonitor()

    def _contract_run_id(self) -> str:
        match = re.fullmatch(r"models:/([^@/]+)@([^/]+)", self.model_uri)
        if not match:
            raise ValueError("Model URI must use models:/<name>@<alias> so its approved contract can be resolved.")
        name, alias = match.groups()
        version = MlflowClient(tracking_uri=self.tracking_uri).get_model_version_by_alias(name, alias)
        if not version.run_id:
            raise ValueError("The champion model version has no source run for its model contract.")
        self.model_version = version.version
        return version.run_id

    def load(self) -> None:
        """Fail closed if model or its separately logged decision contract is absent."""
        mlflow.set_tracking_uri(self.tracking_uri)
        run_id = self._contract_run_id()
        contract_path = Path(MlflowClient(tracking_uri=self.tracking_uri).download_artifacts(run_id, "model_contract.json"))
        self.contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if not self.contract.get("feature_names") or "decision_threshold" not in self.contract:
            raise ValueError("Champion model contract is incomplete.")
        self.model = mlflow.sklearn.load_model(self.model_uri)

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        if self.model is None or self.contract is None:
            self.load()
        assert self.contract is not None
        row = pd.DataFrame([features])
        expected = self.contract["feature_names"]
        missing, unexpected = set(expected) - set(row), set(row) - set(expected)
        if missing or unexpected:
            raise ValueError(f"Feature schema mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        if row.isna().any().any():
            raise ValueError("Missing feature values are not allowed by this model contract.")
        row = row.loc[:, expected]
        score = float(self.model.predict_proba(row)[0][1]) if hasattr(self.model, "predict_proba") else None
        threshold = float(self.contract["decision_threshold"])
        response = {
            "prediction": int(score >= threshold) if score is not None else int(self.model.predict(row)[0]),
            "fraud_probability": score,
            "threshold": threshold,
            "model_uri": self.model_uri,
            "model_version": self.model_version,
            "schema_version": self.contract["schema_version"],
        }
        self.monitor.log_prediction(row.iloc[0].to_dict(), response)
        return response


service = FraudPredictionService()
app = FastAPI(title="Fraud Detection API", version="1.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    try:
        service.load()
        return {"status": "ok", "model_uri": service.model_uri}
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return service.monitor.summary()


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        return service.predict(request.features)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="Prediction service is unavailable.") from error
