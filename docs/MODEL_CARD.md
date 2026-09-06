# Fraud Detection Model Card

## Intended use

Rank credit-card transactions for fraud review. The output is a model score and binary decision; it must support—not replace—fraud-operations review. It is not approved for credit decisions or customer eligibility.

## Data and evaluation

The source is the anonymized ULB credit-card dataset. A stratified 20% original-distribution holdout is isolated before development balancing. Report fraud-class precision, recall, F1, PR-AUC, ROC-AUC, and a confusion matrix. Accuracy is supplementary only.

## Release gate and ownership

An MLflow registry champion requires final-holdout fraud F1 ≥ 0.85 and recall ≥ 0.80. A model owner reviews the MLflow run, artifacts, schema, data lineage, and model card before explicit promotion. Retraining only creates candidates.

## Monitoring and limitations

The API records privacy-minimized inference outputs and aggregate request schema. Once delayed labels are available, compare realized precision/recall, alert on score/feature distribution drift, and investigate false negatives. Dataset features are anonymized, so fairness analysis by protected group is not possible from this data alone.

## Rollback

Use the MLflow Registry to reassign the `champion` alias to the previous approved version. Record the incident, version, reason, and approver in the deployment change log.
