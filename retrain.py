"""Scheduled retraining entry point; promotion stays a human decision."""
from main import FraudDetectionProject
from mlops import MLflowExperimentManager
def main() -> None:
    project = FraudDetectionProject()
    project.load_data(); project.preprocess_data(); project.run_experiments(); project.validate_models()
    print(MLflowExperimentManager().log_project_models(project).round(4))
if __name__ == "__main__": main()
