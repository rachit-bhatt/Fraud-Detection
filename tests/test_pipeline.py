import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

# Add parent directory to path so we can import main
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import FraudDetectionProject, TrainingConfig

def test_metrics_and_grid_metadata(tmp_path):
    frame = pd.DataFrame({"V1": list(range(40)), "V2": list(range(40, 80)), "Class": [0] * 20 + [1] * 20})
    path = tmp_path / "creditcard.csv"
    frame.to_csv(path, index=False)
    project = FraudDetectionProject(TrainingConfig(dataset_path=path, validation_size=.2, test_size=.25, non_fraud_sample_fraction=1, cv_folds=2))
    project.load_data(); project.preprocess_data()
    project.train_model("lr", LogisticRegression(max_iter=200), {"C": [0.1, 1]})
    result = project.evaluate_model("lr")
    assert {"fraud_precision", "fraud_recall", "fraud_f1", "roc_auc", "pr_auc", "confusion_matrix"} <= result.keys()
    assert project.training_metadata["lr"]["best_params"]
