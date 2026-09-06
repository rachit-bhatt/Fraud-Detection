"""Local inference observability; never persists raw transaction values."""
from __future__ import annotations
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

class PredictionMonitor:
    def __init__(self, log_path: str | None = None) -> None:
        self.log_path = Path(log_path or os.getenv("PREDICTION_LOG_PATH", "monitoring/predictions.jsonl"))
    def log_prediction(self, features: dict[str, float], response: dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": datetime.now(UTC).isoformat(), "feature_count": len(features), "prediction": response["prediction"], "fraud_probability": response["fraud_probability"], "model_uri": response["model_uri"]}
        with self.log_path.open("a", encoding="utf-8") as output: output.write(json.dumps(record) + "\n")
    def summary(self) -> dict[str, Any]:
        if not self.log_path.exists(): return {"prediction_count": 0, "fraud_prediction_count": 0}
        records = [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines() if line]
        counts = Counter(record["prediction"] for record in records)
        return {"prediction_count": len(records), "fraud_prediction_count": counts[1], "fraud_prediction_rate": counts[1] / len(records) if records else 0.0}
