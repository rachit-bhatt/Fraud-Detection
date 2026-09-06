"""Reproducible fraud-model training, validation, and MLflow tracking.

Run ``python main.py --quick`` for a local smoke-sized model search, or omit
``--quick`` for the complete configured search. The final validation split is
never used for sampling, hyperparameter search, or threshold selection.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mlops import MLflowExperimentManager


@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: Path = Path("data/creditcard.csv")
    model_path: Path = Path("models/fraud_detection_model.pkl")
    random_state: int = 42
    validation_size: float = 0.20
    test_size: float = 0.30
    non_fraud_sample_fraction: float = 0.10
    cv_folds: int = 5
    search_scoring: str = "f1"
    threshold_candidates: tuple[float, ...] = tuple(np.round(np.arange(0.10, 0.91, 0.05), 2))


class FraudDetectionProject:
    """Owns data preparation, candidate training, evaluation, and selection."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.df_original: pd.DataFrame | None = None
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.X_validation: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.y_validation: pd.Series | None = None
        self.models: dict[str, ClassifierMixin] = {}
        self.training_metadata: dict[str, dict[str, Any]] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.validation_results: dict[str, dict[str, Any]] = {}
        self.thresholds: dict[str, float] = {}
        self.best_model: str | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns) if self.X_train is not None else []

    def load_data(self) -> None:
        if not self.config.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.config.dataset_path}")
        self.df_original = pd.read_csv(self.config.dataset_path)
        if "Class" not in self.df_original or set(self.df_original.Class.dropna().unique()) - {0, 1}:
            raise ValueError("Dataset requires a binary 'Class' target column.")

    def preprocess_data(self) -> None:
        """Create an untouched holdout, then balance development data only."""
        if self.df_original is None:
            raise ValueError("Run load_data() before preprocess_data().")
        X, y = self.df_original.drop(columns="Class"), self.df_original.Class
        development_X, self.X_validation, development_y, self.y_validation = train_test_split(
            X, y, test_size=self.config.validation_size, random_state=self.config.random_state, stratify=y
        )
        development = development_X.assign(Class=development_y)
        fraud = development[development.Class == 1]
        non_fraud = development[development.Class == 0].sample(
            frac=self.config.non_fraud_sample_fraction, random_state=self.config.random_state
        )
        balanced = pd.concat([fraud, non_fraud], ignore_index=True).sample(frac=1, random_state=self.config.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            balanced.drop(columns="Class"), balanced.Class, test_size=self.config.test_size,
            random_state=self.config.random_state, stratify=balanced.Class,
        )

    def train_model(self, model_name: str, model: ClassifierMixin, param_grid: dict[str, list[Any]] | None = None) -> None:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Run preprocess_data() before training.")
        metadata: dict[str, Any] = {"search_performed": bool(param_grid), "model_class": type(model).__name__}
        if param_grid:
            search = GridSearchCV(model, param_grid, cv=self.config.cv_folds, n_jobs=-1,
                                  scoring=self.config.search_scoring, refit=True, verbose=1).fit(self.X_train, self.y_train)
            self.models[model_name] = search.best_estimator_
            metadata.update({"cv_folds": self.config.cv_folds, "search_scoring": self.config.search_scoring,
                             "best_params": search.best_params_, "best_cv_score": float(search.best_score_),
                             "candidate_count": len(search.cv_results_["params"])})
        else:
            self.models[model_name] = model.fit(self.X_train, self.y_train)
        self.training_metadata[model_name] = metadata

    @staticmethod
    def _prediction_scores(model: ClassifierMixin, X: pd.DataFrame) -> np.ndarray | None:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
        return None

    @staticmethod
    def _metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, Any]:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return {"accuracy": float(report["accuracy"]), "fraud_precision": float(report["1"]["precision"]),
                "fraud_recall": float(report["1"]["recall"]), "fraud_f1": float(report["1"]["f1-score"]),
                "roc_auc": float(roc_auc_score(y_true, y_score)) if y_score is not None else float("nan"),
                "pr_auc": float(average_precision_score(y_true, y_score)) if y_score is not None else float("nan"),
                "confusion_matrix": confusion_matrix(y_true, y_pred), "classification_report": report}

    def evaluate_model(self, model_name: str, threshold: float = 0.5) -> dict[str, Any]:
        if self.X_test is None or self.y_test is None:
            raise ValueError("Run preprocess_data() before evaluation.")
        model = self.models[model_name]
        scores = self._prediction_scores(model, self.X_test)
        predictions = (scores >= threshold).astype(int) if scores is not None else model.predict(self.X_test)
        result = self._metrics(self.y_test, predictions, scores)
        result["threshold"] = threshold
        self.results[model_name] = result
        return result

    def tune_threshold(self, model_name: str, metric: str = "fraud_f1") -> float:
        """Choose a score threshold on development data; never final holdout."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Run preprocess_data() before threshold tuning.")
        scores = self._prediction_scores(self.models[model_name], self.X_test)
        if scores is None:
            self.thresholds[model_name] = 0.5
            return 0.5
        best_threshold, best_value = 0.5, -1.0
        for threshold in self.config.threshold_candidates:
            value = self._metrics(self.y_test, (scores >= threshold).astype(int), scores)[metric]
            if value > best_value:
                best_threshold, best_value = threshold, value
        self.thresholds[model_name] = float(best_threshold)
        return float(best_threshold)

    def validate_models(self) -> pd.DataFrame:
        """Evaluate each candidate once on original-distribution final holdout."""
        if self.X_validation is None or self.y_validation is None:
            raise ValueError("Run preprocess_data() before final validation.")
        for name, model in self.models.items():
            scores = self._prediction_scores(model, self.X_validation)
            threshold = self.thresholds.get(name, 0.5)
            predictions = (scores >= threshold).astype(int) if scores is not None else model.predict(self.X_validation)
            result = self._metrics(self.y_validation, predictions, scores)
            result["threshold"] = threshold
            self.validation_results[name] = result
        return self.summary(final=True)

    def summary(self, final: bool = False) -> pd.DataFrame:
        source = self.validation_results if final else self.results
        rows = [{"model": name, **{k: v for k, v in result.items() if isinstance(v, (int, float, np.floating))}}
                for name, result in source.items()]
        return pd.DataFrame(rows).set_index("model").sort_values("fraud_f1", ascending=False)

    def select_best_model(self, metric: str = "fraud_f1") -> str:
        summary = self.summary(final=True)
        if summary.empty or metric not in summary:
            raise ValueError("Run validate_models() before selecting a model.")
        self.best_model = str(summary[metric].idxmax())
        return self.best_model

    def save_model(self, model_name: str | None = None) -> Path:
        name = model_name or self.best_model
        if name is None or name not in self.models:
            raise ValueError("Select or supply a trained model before saving.")
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models[name], self.config.model_path)
        return self.config.model_path

    def run_experiments(self, quick: bool = False) -> None:
        """Train existing five model families with reproducible bounded searches."""
        rf_grid = {"n_estimators": [100] if quick else [100, 200], "max_depth": [10, 20], "min_samples_split": [2, 5]}
        gb_grid = {"n_estimators": [100] if quick else [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
        svm_grid = {"model__C": [1] if quick else [0.1, 1, 10], "model__kernel": ["linear", "rbf"]}
        knn_grid = {"model__n_neighbors": [5] if quick else [3, 5, 7], "model__weights": ["uniform", "distance"]}
        candidates: list[tuple[str, ClassifierMixin, dict[str, list[Any]] | None]] = [
            ("RandomForest", RandomForestClassifier(random_state=self.config.random_state, n_jobs=-1), rf_grid),
            ("GradientBoosting", GradientBoostingClassifier(random_state=self.config.random_state), gb_grid),
            ("LogisticRegression", Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=self.config.random_state))]), None),
            ("SVM", Pipeline([("scale", StandardScaler()), ("model", SVC(probability=True, random_state=self.config.random_state))]), svm_grid),
            ("KNN", Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())]), knn_grid),
        ]
        for name, model, grid in candidates:
            self.train_model(name, model, grid)
            self.evaluate_model(name)
            self.tune_threshold(name)
            self.evaluate_model(name, self.thresholds[name])

    def metadata(self) -> dict[str, Any]:
        if self.df_original is None:
            return {}
        return {"dataset_path": str(self.config.dataset_path), "dataset_rows": len(self.df_original),
                "fraud_rate": float(self.df_original.Class.mean()), "feature_count": len(self.feature_names),
                "random_state": self.config.random_state, "validation_size": self.config.validation_size,
                "development_non_fraud_fraction": self.config.non_fraud_sample_fraction}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use reduced parameter grids for a smoke-sized run.")
    parser.add_argument("--promote", action="store_true", help="Attempt gated registry promotion after validation.")
    args = parser.parse_args()
    project = FraudDetectionProject()
    project.load_data(); project.preprocess_data(); project.run_experiments(quick=args.quick)
    print("Development metrics:\n", project.summary().round(4))
    print("Final holdout metrics:\n", project.validate_models().round(4))
    tracker = MLflowExperimentManager()
    print("MLflow runs:\n", tracker.log_project_models(project).round(4))
    winner = project.select_best_model()
    print(f"Best final-holdout candidate: {winner}")
    if args.promote:
        print(json.dumps(tracker.register_best_model(project), indent=2, default=str))


if __name__ == "__main__":
    main()
