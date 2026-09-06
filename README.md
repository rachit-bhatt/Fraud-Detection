# Fraud Detection — Production-Oriented MLOps Portfolio

This project trains fraud classifiers on the ULB credit-card dataset and demonstrates reproducible experiments, MLflow tracking/registry gates, a FastAPI service, a container, CI, prediction telemetry, and an Azure template.

## Architecture

`data/creditcard.csv` → `main.py` → MLflow **Fraud Detection** experiment → gated Model Registry → `api.py` → prediction telemetry.

Training makes a stratified 20% **untouched original-distribution** final holdout before balancing development data. Searches and threshold tuning use development data only. This avoids being misled by accuracy on an extremely imbalanced fraud problem.

Every candidate records fraud precision, recall, F1, ROC-AUC, PR-AUC, a confusion matrix, explicit GridSearchCV `best_params_`, search score, JSON evaluation artifact, model signature, and serialized model. Fraud F1/recall are the key comparison and promotion metrics.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --quick
```

Remove `--quick` for the full bounded grids. Both modes train Random Forest, Gradient Boosting, Logistic Regression, SVM, and KNN.

Open MLflow in another terminal:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`, choose **Fraud Detection**, sort by `final_fraud_f1`/`final_fraud_recall`, and inspect Parameters and Artifacts. The lower-case experiment is historical; new runs use the named experiment above.

## Registry, API, and operations

`python main.py --quick --promote` promotes only after final-holdout F1 ≥ 0.85 and recall ≥ 0.80. Failed candidates do not replace the champion. `retrain.py` creates candidates but never auto-promotes them.

```powershell
uvicorn api:app --reload
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

The API defaults to `models:/fraud-detection-model@champion`; configure `MLFLOW_TRACKING_URI` and `MLFLOW_MODEL_URI` externally. It provides `/health`, `/predict`, and `/metrics`. Telemetry retains outputs and aggregate schema information, never raw transaction values.

GitHub Actions tests and container-builds each push/PR. `azure/containerapp.yaml` is a credential-free Azure Container Apps template; set ACR image, managed MLflow URI, identities, and secrets in Azure/CI.

## Interview points

- PR-AUC and fraud recall are more meaningful than accuracy under severe imbalance.
- Final-holdout isolation prevents threshold/search optimism from becoming deployment risk.
- MLflow runs, artifacts, registry gates, and explicit promotion create traceability.
- Schema validation and privacy-aware telemetry are production inference controls.
