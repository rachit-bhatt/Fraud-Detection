# Import necessary libraries
import os
import requests
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

class FraudDetectionProject:
    def __init__(self, config):
        """
        Initialize the project with a configuration dictionary.
        """
        self.dataset_url = config['dataset_url']
        self.dataset_path = config['dataset_path']
        self.model_path = config['model_path']
        self.model_type = config['model_type']
        self.spark = None
        self.df_spark = None
        self.df_pandas = None
        self.model = None
    
    def download_data(self):
        """
        Download the dataset if it does not exist locally.
        """
        if not os.path.exists(self.dataset_path):
            print("Downloading dataset...")
            response = requests.get(self.dataset_url)
            with open(self.dataset_path, 'wb') as f:
                f.write(response.content)
            print(f"Dataset downloaded at {self.dataset_path}")
        else:
            print("Dataset already exists.")
    
    def initialize_spark(self):
        """
        Initialize a Spark session.
        """
        print("Initializing Spark session...")
        self.spark = SparkSession.builder.appName("FraudDetection").getOrCreate()
        print("Spark session initialized.")
    
    def load_data_with_spark(self):
        """
        Load the dataset using PySpark.
        """
        print("Loading dataset using Spark...")
        self.df_spark = self.spark.read.csv(self.dataset_path, header=True, inferSchema=True)
        print(f"Dataset loaded with {self.df_spark.count()} rows and {len(self.df_spark.columns)} columns.")
    
    def preprocess_data_with_spark(self):
        """
        Preprocess the data using Spark: Convert to Pandas for modeling.
        """
        print("Preprocessing dataset with Spark...")
        self.df_spark = self.df_spark.withColumn("Class", col("Class").cast("integer"))
        
        # Downsample data for training purposes (optional depending on data size)
        fraud_cases = self.df_spark.filter(self.df_spark['Class'] == 1)
        non_fraud_cases = self.df_spark.filter(self.df_spark['Class'] == 0).sample(fraction=0.1)
        
        balanced_data = fraud_cases.union(non_fraud_cases)
        print(f"Balanced dataset with {balanced_data.count()} rows.")
        
        # Convert Spark DataFrame to Pandas for training
        self.df_pandas = balanced_data.toPandas()
        print("Converted Spark DataFrame to Pandas.")
    
    def preprocess_data(self):
        """
        Split data into features and labels, then into training and test sets.
        """
        print("Splitting dataset into training and test sets...")
        X = self.df_pandas.drop(columns=['Class'])  # Features
        y = self.df_pandas['Class']  # Target (fraud label)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print("Data split completed.")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train a RandomForest model (or another model type specified).
        """
        print(f"Training {self.model_type} model...")
        
        if self.model_type == 'RandomForest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise NotImplementedError(f"Model {self.model_type} is not implemented.")
        
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved at {self.model_path}")
    
    def load_model(self):
        """
        Load the trained model from file.
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            print("Model not found. Please train the model first.")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on the test data and print out classification report and confusion matrix.
        """
        if self.model is None:
            print("Model is not loaded. Please load or train the model first.")
            return
        
        print("Making predictions and evaluating the model...")
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

# Example configuration dictionary
config = {
    'dataset_url': 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download?resource=download',
    'dataset_path': 'data/creditcard.csv',
    'model_path': 'models/fraud_detection_model.pkl',
    'model_type': 'RandomForest'
}

# Initialize the FraudDetectionProject class with the config
fraud_project = FraudDetectionProject(config)

# Download the dataset
# fraud_project.download_data()

# Initialize Spark session
fraud_project.initialize_spark()

# Load the dataset using Spark
fraud_project.load_data_with_spark()

# Preprocess data using Spark and convert it to Pandas for modeling
fraud_project.preprocess_data_with_spark()

# Split data into training and test sets
X_train, X_test, y_train, y_test = fraud_project.preprocess_data()

# Train the model
fraud_project.train_model(X_train, y_train)

# Evaluate the model
fraud_project.evaluate_model(X_test, y_test)