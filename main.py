# Required Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from mlops import MLflowExperimentManager
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Import PySpark only if Spark is used
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
except ImportError:
    SparkSession = None

# Fraud Detection Project Class
class FraudDetectionProject:
    def __init__(self, use_local=False, mlflow_manager=None):
        self.dataset_url = 'https://www.kaggle.com/mlg-ulb/creditcardfraud/download'
        self.dataset_path = 'data/creditcard.csv'
        self.model_path = 'models/fraud_detection_model.pkl'
        self.use_local = use_local
        self.spark = None
        self.df_spark = None
        self.df_pandas = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.mlflow_manager = mlflow_manager or MLflowExperimentManager()

        if not self.use_local and SparkSession is not None:
            self.spark = SparkSession.builder.appName("FraudDetection").getOrCreate()
    
    def load_data(self):
        if not self.use_local and self.spark is not None:
            # Read dataset into a Spark DataFrame
            self.df_spark = self.spark.read.csv(self.dataset_path, header=True, inferSchema=True)
        else:
            # Load the dataset locally with Pandas
            self.df_pandas = pd.read_csv(self.dataset_path)
    
    def preprocess_data(self):
        if not self.use_local and self.spark is not None:
            # Spark Preprocessing
            self.df_spark = self.df_spark.withColumn("Class", col("Class").cast("integer"))
            
            # Balance the dataset
            fraud_cases = self.df_spark.filter(self.df_spark['Class'] == 1)
            non_fraud_cases = self.df_spark.filter(self.df_spark['Class'] == 0).sample(fraction=0.1)
            balanced_data = fraud_cases.union(non_fraud_cases)
            
            # Convert Spark DataFrame to Pandas DataFrame
            self.df_pandas = balanced_data.toPandas()
        else:
            # Local Pandas Preprocessing
            # Balance the dataset locally
            fraud_cases = self.df_pandas[self.df_pandas['Class'] == 1]
            non_fraud_cases = self.df_pandas[self.df_pandas['Class'] == 0].sample(frac=0.1, random_state=42)
            self.df_pandas = pd.concat([fraud_cases, non_fraud_cases])

        # Split data into X (features) and y (target)
        X = self.df_pandas.drop(columns=['Class'])
        y = self.df_pandas['Class']
        
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )
    
    def train_model(self, model_name, model, param_grid=None):
        # Train a model with optional hyper-parameter tuning
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            self.models[model_name] = best_model
        else:
            model.fit(self.X_train, self.y_train)
            self.models[model_name] = model

    @staticmethod
    def _prediction_scores(model, X):
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, 'decision_function'):
            return model.decision_function(X)
        return None
    
    def evaluate_model(self, model_name):
        # Evaluate using fraud-focused metrics and create the MLflow run.
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        y_score = self._prediction_scores(model, self.X_test)
        report = classification_report(
            self.y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
        
        self.results[model_name] = {
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred, zero_division=0),
            'accuracy': report['accuracy'],
            'accuracy_score': accuracy_score(self.y_test, y_pred),
            'fraud_precision': report['1']['precision'],
            'fraud_recall': report['1']['recall'],
            'fraud_f1': report['1']['f1-score'],
            'roc_auc': roc_auc_score(self.y_test, y_score) if y_score is not None else np.nan,
            'pr_auc': average_precision_score(self.y_test, y_score) if y_score is not None else np.nan,
            'y_pred': y_pred,
        }
        self.mlflow_manager.log_model_run(self, model_name)
    
    def print_results(self, model_name):
        result = self.results[model_name]
        print(f"Results for {model_name}:\n")
        print(f"Confusion Matrix:\n{result['confusion_matrix']}\n")
        print(f"Classification Report:\n{result['classification_report']}\n")
        print(f"Fraud Precision: {result['fraud_precision']:.4f}")
        print(f"Fraud Recall: {result['fraud_recall']:.4f}")
        print(f"Fraud F1: {result['fraud_f1']:.4f}")
        print(f"PR-AUC: {result['pr_auc']:.4f}")
        print(f"ROC-AUC: {result['roc_auc']:.4f}\n")
    
    def visualize_confusion_matrix(self, model_name):
        # Visualize the confusion matrix for the specified model
        matrix = self.results[model_name]['confusion_matrix']
        plt.figure(figsize=(25.6, 16))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix for {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def visualize_correlation_heatmap(self):
        # Visualize a heatmap of correlations between features
        correlation_matrix = self.df_pandas.corr()
        plt.figure(figsize=(25.6, 16))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap")
        plt.show()
    
    def run_experiments(self):
        # Random Forest with Hyper-Parameter Tuning
        param_grid_rf = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }
        self.train_model("RandomForest", RandomForestClassifier(), param_grid_rf)
        self.evaluate_model("RandomForest")
        
        # Gradient Boosting (XGBoost) with Hyper-Parameter Tuning
        param_grid_gb = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
        self.train_model("GradientBoosting", GradientBoostingClassifier(), param_grid_gb)
        self.evaluate_model("GradientBoosting")
        
        # Logistic Regression (Standard)
        self.train_model("LogisticRegression", LogisticRegression(max_iter=1000))
        self.evaluate_model("LogisticRegression")
        
        # Support Vector Machine with Hyper-Parameter Tuning
        param_grid_svm = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
        self.train_model("SVM", SVC(), param_grid_svm)
        self.evaluate_model("SVM")
        
        # K-Nearest Neighbors with Hyper-Parameter Tuning
        param_grid_knn = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
        self.train_model("KNN", KNeighborsClassifier(), param_grid_knn)
        self.evaluate_model("KNN")
    
    def visualize_all_results(self):
        for model_name in self.models.keys():
            self.print_results(model_name)
            self.visualize_confusion_matrix(model_name)
    
    def compare_models(self):
        # Compare fraud metrics; accuracy is retained only as a reference.
        model_names = list(self.results.keys())
        metrics = pd.DataFrame({
            'Fraud F1': [self.results[model]['fraud_f1'] for model in model_names],
            'Fraud Recall': [self.results[model]['fraud_recall'] for model in model_names],
            'PR-AUC': [self.results[model]['pr_auc'] for model in model_names],
            'Accuracy': [self.results[model]['accuracy'] for model in model_names],
        }, index=model_names)
        print(metrics.sort_values('Fraud F1', ascending=False).round(4))

        fig = go.Figure(data=[
            go.Bar(name='Fraud F1', x=model_names, y=metrics['Fraud F1']),
            go.Bar(name='Fraud Recall', x=model_names, y=metrics['Fraud Recall']),
            go.Bar(name='PR-AUC', x=model_names, y=metrics['PR-AUC']),
        ])
        fig.update_layout(
            title='Fraud Model Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            template='plotly_white'
        )
        fig.show()

    def compare_accuracies(self):
        """Backward-compatible alias for the fraud-metrics comparison."""
        self.compare_models()

# Main Function to run the project
if __name__ == "__main__":
    # Create an instance of the project with the `use_local` flag
    # Set use_local=True to run locally, use_local=False to run with Spark
    fraud_project = FraudDetectionProject(use_local=True)

    # Load data and initialize
    fraud_project.load_data()

    # Preprocess the data and split into training and testing sets
    fraud_project.preprocess_data()

    # Run experiments with various algorithms
    fraud_project.run_experiments()

    # Visualize correlation heatmap and confusion matrices for all models
    fraud_project.visualize_correlation_heatmap()
    fraud_project.visualize_all_results()

    # Compare fraud-focused metrics and display the MLflow experiment runs
    fraud_project.compare_models()
    display(fraud_project.mlflow_manager.tracked_run_summary().round(4))