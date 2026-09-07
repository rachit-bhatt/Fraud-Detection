import pandas as pd
from sklearn.linear_model import LogisticRegression

from main import FraudDetectionProject, TrainingConfig
from mlops import MLflowExperimentManager
from api import FraudPredictionService

def test_metrics_and_grid_metadata(tmp_path):
    frame = pd.DataFrame({"V1": list(range(40)), "V2": list(range(40, 80)), "Class": [0] * 20 + [1] * 20})
    path = tmp_path / "creditcard.csv"
    frame.to_csv(path, index=False)
    project = FraudDetectionProject(TrainingConfig(dataset_path=path, validation_size=.2, threshold_validation_size=.25, non_fraud_sample_fraction=1, cv_folds=2))
    project.load_data(); project.preprocess_data()
    project.train_model("lr", LogisticRegression(max_iter=200), {"C": [0.1, 1]})
    result = project.evaluate_model("lr")
    assert {"fraud_precision", "fraud_recall", "fraud_f1", "roc_auc", "pr_auc", "confusion_matrix"} <= result.keys()
    assert project.training_metadata["lr"]["best_params"]
    project.tune_threshold("lr")
    project.evaluate_model("lr", project.thresholds["lr"])
    project.validate_models()
    contract = project.model_contract("lr")
    assert contract["decision_threshold"] == project.thresholds["lr"]
    tracker = MLflowExperimentManager(
        tracking_uri="sqlite:///" + str(tmp_path / "mlflow.db"),
        artifact_root=str(tmp_path / "artifacts"),
        experiment_name="contract-test",
    )
    tracker.log_project_models(project)
    run = tracker.client.get_run(tracker.run_ids["lr"])
    assert "model_contract.json" in [item.path for item in tracker.client.list_artifacts(run.info.run_id)]
    promotion = tracker.register_best_model(project)
    service = FraudPredictionService(promotion["model_uri"])
    service.tracking_uri = tracker.tracking_uri
    prediction = service.predict(project.X_validation.iloc[0].astype(float).to_dict())
    assert prediction["threshold"] == contract["decision_threshold"]
