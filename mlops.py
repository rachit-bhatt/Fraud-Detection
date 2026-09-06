"""MLflow experiment tracking and gated model-registry operations."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature


class MLflowExperimentManager:
    """Log one fully evaluated candidate per MLflow run.

    Registry promotion is explicit and protected by final-holdout gates; normal
    experiment tracking never changes the deployed model.
    """

    def __init__(self, tracking_uri: str | None = None, experiment_name: str = "Fraud Detection",
                 registered_model_name: str = "fraud-detection-model", artifact_root: str = "mlruns",
                 min_fraud_f1: float = 0.85, min_fraud_recall: float = 0.80) -> None:
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        self.experiment_name, self.registered_model_name = experiment_name, registered_model_name
        self.artifact_root = Path(artifact_root).resolve()
        self.min_fraud_f1, self.min_fraud_recall = min_fraud_f1, min_fraud_recall
        self.run_ids: dict[str, str] = {}
        self.model_uris: dict[str, str] = {}
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment = mlflow.get_experiment(mlflow.create_experiment(experiment_name, artifact_location=self.artifact_root.as_uri()))
        self.experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    @staticmethod
    def _safe_value(value: Any) -> str:
        return (json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list, tuple)) else str(value))[:500]

    def _parameters(self, project: Any, model_name: str) -> dict[str, str]:
        params = {f"estimator__{key}": self._safe_value(value)
                  for key, value in project.models[model_name].get_params(deep=False).items()}
        metadata = project.training_metadata[model_name]
        for key in ("cv_folds", "search_scoring", "candidate_count", "best_cv_score"):
            if key in metadata:
                params[f"grid_search__{key}"] = self._safe_value(metadata[key])
        for key, value in metadata.get("best_params", {}).items():
            params[f"grid_search__best__{key}"] = self._safe_value(value)
        return params

    @staticmethod
    def _numeric_metrics(result: dict[str, Any], prefix: str) -> dict[str, float]:
        names = ("accuracy", "fraud_precision", "fraud_recall", "fraud_f1", "roc_auc", "pr_auc", "threshold")
        return {f"{prefix}{name}": float(result[name]) for name in names if name in result and np.isfinite(result[name])}

    def _confusion_matrix_artifact(self, model_name: str, matrix: np.ndarray) -> Path:
        path = self.artifact_root / f"{model_name}_confusion_matrix.png"
        fig, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        fig.colorbar(image, ax=axis)
        axis.set(title=f"{model_name} confusion matrix", xlabel="Predicted class", ylabel="Actual class", xticks=[0, 1], yticks=[0, 1])
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        return path

    @staticmethod
    def _serializable_result(result: dict[str, Any]) -> dict[str, Any]:
        return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items()}

    def log_project_models(self, project: Any) -> pd.DataFrame:
        """Track candidates after final validation, one MLflow run per model."""
        if not project.validation_results:
            raise ValueError("Run final validation before MLflow logging.")
        rows = []
        for model_name, model in project.models.items():
            development, final = project.results[model_name], project.validation_results[model_name]
            metrics = self._numeric_metrics(development, "development_") | self._numeric_metrics(final, "final_")
            tags = {"model_name": model_name, "model_class": type(model).__name__, "task": "binary_fraud_detection",
                    "evaluation_strategy": "balanced_development_plus_untouched_original_holdout",
                    "search_performed": str(project.training_metadata[model_name]["search_performed"])}
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=model_name) as run:
                mlflow.set_tags(tags); mlflow.log_params(self._parameters(project, model_name)); mlflow.log_metrics(metrics)
                mlflow.log_dict({"project_metadata": project.metadata(), "training_metadata": project.training_metadata[model_name],
                                 "development": self._serializable_result(development), "final_holdout": self._serializable_result(final)},
                                "evaluation/evaluation.json")
                mlflow.log_artifact(str(self._confusion_matrix_artifact(model_name, final["confusion_matrix"])), "evaluation")
                sample = project.X_train.head(5)
                info = mlflow.sklearn.log_model(model, name="model", signature=infer_signature(sample, model.predict(sample)),
                                                input_example=sample.head(2))
                self.run_ids[model_name], self.model_uris[model_name] = run.info.run_id, info.model_uri
            rows.append({"model": model_name, "run_id": self.run_ids[model_name], **metrics})
        return pd.DataFrame(rows).set_index("model").sort_values("final_fraud_f1", ascending=False)

    def tracked_run_summary(self) -> pd.DataFrame:
        rows = [{"model": name, "run_id": run_id, **self.client.get_run(run_id).data.metrics}
                for name, run_id in self.run_ids.items()]
        return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()

    def register_best_model(self, project: Any, metric: str = "fraud_f1") -> dict[str, Any]:
        """Register and alias a candidate only if final-holdout fraud gates pass."""
        winner, final = project.select_best_model(metric), project.validation_results[project.best_model]
        if final["fraud_f1"] < self.min_fraud_f1 or final["fraud_recall"] < self.min_fraud_recall:
            raise ValueError(f"Promotion blocked: final fraud_f1={final['fraud_f1']:.4f} (minimum {self.min_fraud_f1}), final fraud_recall={final['fraud_recall']:.4f} (minimum {self.min_fraud_recall}).")
        if winner not in self.run_ids:
            raise ValueError("Log MLflow runs before registry promotion.")
        version = mlflow.register_model(self.model_uris[winner], self.registered_model_name)
        self.client.set_model_version_tag(self.registered_model_name, version.version, "threshold", str(final["threshold"]))
        self.client.set_model_version_tag(self.registered_model_name, version.version, "promotion_gate", "final_holdout_fraud_f1_and_recall")
        self.client.set_registered_model_alias(self.registered_model_name, "champion", version.version)
        return {"model": winner, "run_id": self.run_ids[winner], "version": version.version,
                "model_uri": f"models:/{self.registered_model_name}@champion", "final_metrics": self._serializable_result(final)}
