import numpy as np

from api import FraudPredictionService
from monitoring import PredictionMonitor


class FixedScoreModel:
    def predict_proba(self, _row):
        return np.array([[0.70, 0.30]])


def test_prediction_uses_approved_contract_threshold(tmp_path):
    service = FraudPredictionService("models:/example@champion")
    service.model = FixedScoreModel()
    service.contract = {
        "schema_version": 1,
        "feature_names": ["V1", "Amount"],
        "decision_threshold": 0.25,
    }
    service.model_version = "7"
    service.monitor = PredictionMonitor(str(tmp_path / "predictions.jsonl"))

    response = service.predict({"V1": 1.0, "Amount": 10.0})

    assert response["fraud_probability"] == 0.30
    assert response["threshold"] == 0.25
    assert response["prediction"] == 1
