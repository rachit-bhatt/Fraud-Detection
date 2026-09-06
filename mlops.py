from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient


class MLflowExperimentManager:
    """Track trained FraudDetectionProject models and manage their registry alias."""

    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlflow.db",
        experiment_name: str = "fraud-detection",
        registered_model_name: str = "fraud-detection-model",
        artifact_root: str = "mlruns",
        min_fraud_f1: float = 0.85,
        min_fraud_recall: float = 0.80,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.registered_model_name = registered_model_name
        self.artifact_root = Path(artifact_root)
        self.min_fraud_f1 = min_fraud_f1
        self.min_fraud_recall = min_fraud_recall
        self.run_ids: dict[str, str] = {}

        self.artifact_root.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    @staticmethod
    def _safe_params(model: Any) -> dict[str, str]:
        params = {}
        for name, value in model.get_params(deep=False).items():
            value = str(value)
            params[name] = value[:250]
        return params

    @staticmethod
    def _metrics(project: Any, model_name: str) -> dict[str, float]:
        report = project.summarize_model_metrics().loc[model_name]
        return {
            "accuracy": float(report["accuracy"]),
            "fraud_precision": float(report["fraud_precision"]),
            "fraud_recall": float(report["fraud_recall"]),
            "fraud_f1": float(report["fraud_f1"]),
            "roc_auc": float(report["roc_auc"]),
        }

    def _log_confusion_matrix(self, project: Any, model_name: str) -> str:
        output_path = self.artifact_root / f"{model_name}_confusion_matrix.png"
        matrix = project.results[model_name]["confusion_matrix"]
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{model_name} Confusion Matrix")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, matrix[row, column], ha="center", va="center")
        figure.tight_layout()
        figure.savefig(output_path)
        plt.close(figure)
        return str(output_path)

    def log_project_models(self, project: Any) -> pd.DataFrame:
        """Create one MLflow run per trained model and log metrics/artifacts."""
        records = []

        for model_name, model in project.models.items():
            metrics = self._metrics(project, model_name)
            with mlflow.start_run(run_name=model_name) as run:
                mlflow.set_tags({
                    "model_name": model_name,
                    "task": "fraud_detection",
                    "data_path": project.dataset_path,
                })
                mlflow.log_params(self._safe_params(model))
                mlflow.log_metrics(metrics)
                mlflow.log_dict({
                    "metrics": metrics,
                    "confusion_matrix": project.results[model_name]["confusion_matrix"].tolist(),
                }, "evaluation.json")
                mlflow.log_artifact(self._log_confusion_matrix(project, model_name))
                mlflow.sklearn.log_model(model, name="model")
                self.run_ids[model_name] = run.info.run_id

            records.append({"model": model_name, "run_id": self.run_ids[model_name], **metrics})

        return pd.DataFrame(records).set_index("model").sort_values("fraud_f1", ascending=False)

    def _passes_promotion_gate(self, metrics: pd.Series) -> bool:
        return (
            metrics["fraud_f1"] >= self.min_fraud_f1
            and metrics["fraud_recall"] >= self.min_fraud_recall
        )

    def register_best_model(self, project: Any, metric: str = "fraud_f1") -> dict[str, Any]:
        """Register the best logged model and assign the champion alias if it passes the gate."""
        summary = project.summarize_model_metrics()
        model_name = project.select_best_model(summary, metric=metric)
        metrics = summary.loc[model_name]
        run_id = self.run_ids.get(model_name)
        if run_id is None:
            raise ValueError(f"Model '{model_name}' has not been logged to MLflow.")
        if not self._passes_promotion_gate(metrics):
            raise ValueError(
                f"{model_name} failed promotion gate: fraud_f1={metrics['fraud_f1']:.4f}, "
                f"fraud_recall={metrics['fraud_recall']:.4f}"
            )

        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(model_uri, self.registered_model_name)
        self.client.set_model_version_tag(
            self.registered_model_name,
            model_version.version,
            "promotion_reason",
            f"Passed fraud_f1 >= {self.min_fraud_f1} and fraud_recall >= {self.min_fraud_recall}",
        )
        self.client.set_registered_model_alias(
            self.registered_model_name,
            "champion",
            model_version.version,
        )
        return {
            "model": model_name,
            "run_id": run_id,
            "version": model_version.version,
            "model_uri": f"models:/{self.registered_model_name}@champion",
            "metrics": metrics.to_dict(),
        }

    def champion_uri(self) -> str:
        return f"models:/{self.registered_model_name}@champion"


if __name__ == "__main__":
    print("Use MLflowExperimentManager from the notebook or training application.")
