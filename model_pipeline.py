"""
Water Potability Classification Pipeline
Author: Ben Aissa Amen Allah
Class: 4DS8
Algorithm: Naive Bayes (GaussianNB)
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import os
import mlflow
import mlflow.sklearn
from elasticsearch import Elasticsearch
from datetime import datetime
import time
import json
import matplotlib.pyplot as plt
import numpy as np

# Configure MLflow to use SQLite instead of filesystem store
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ============================================
# ELASTICSEARCH CONNECTION
# ============================================
def get_elasticsearch_client():
    """
    Create and return Elasticsearch client connection
    
    Returns:
        Elasticsearch: Connected ES client or None if connection fails
    """
    try:
        es = Elasticsearch(
            hosts=["http://localhost:9200"],
            verify_certs=False,
            ssl_show_warn=False
        )
        if es.ping():
            print("Connected to Elasticsearch!")
            return es
        else:
            print("Failed to connect to Elasticsearch")
            return None
    except Exception as e:
        print(f"⚠️ Elasticsearch connection error: {e}")
        return None

# Initialize global ES client
es_client = get_elasticsearch_client()

def log_to_elasticsearch(run_id, metrics, params, tags=None, model_name="water_potability"):
    """
    Send MLflow run data to Elasticsearch
    
    Args:
        run_id (str): MLflow run ID
        metrics (dict): Model metrics (accuracy, precision, etc.)
        params (dict): Model parameters and hyperparameters
        tags (dict): Additional tags for the run
        model_name (str): Name of the model
    
    Returns:
        bool: True if successfully logged, False otherwise
    """
    if es_client is None:
        print("⚠️ Elasticsearch not connected, skipping logging")
        return False
    
    try:
        document = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "metrics": metrics,
            "params": params,
            "tags": tags or {}
        }
        
        response = es_client.index(
            index="mlflow-metrics",
            document=document
        )
        
        print(f"Logged to Elasticsearch: {response['result']}")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to log to Elasticsearch: {e}")
        return False


def prepare_data(dataset_path, 
                 missing_value_strategy="median",
                 engineer_features=True,
                 outlier_method="IQR",
                 outlier_threshold=1.5,
                 outlier_columns=None):
    """
    Load and preprocess the water potability dataset.

    Performs the following steps:
    1. Loads CSV data
    2. Fills missing values based on strategy
    3. Engineers features (optional)
    4. Caps outliers based on method and threshold

    Args:
        dataset_path (str): Path to the water potability CSV file
        missing_value_strategy (str): Strategy for handling missing values - "median", "mean", or "drop"
        engineer_features (bool): Whether to create engineered features
        outlier_method (str): Method for outlier detection - "IQR" or "zscore"
        outlier_threshold (float): Threshold for outlier capping (1.5 for IQR, 3.0 for z-score typically)
        outlier_columns (list): Specific columns to cap outliers. If None, uses default set.

    Returns:
        tuple: (pd.DataFrame, list) - Processed dataframe and list of engineered feature names
    """
    print("\n")
    print("Reading dataset")
    df = pd.read_csv(dataset_path)
    print("Processing dataset")
    
    # Handle missing values
    print(f"Handling missing values with strategy: {missing_value_strategy}")
    if missing_value_strategy == "median":
        df.fillna(df.median(), inplace=True)
    elif missing_value_strategy == "mean":
        df.fillna(df.mean(), inplace=True)
    elif missing_value_strategy == "drop":
        df.dropna(inplace=True)
    else:
        raise ValueError(f"Unknown missing value strategy: {missing_value_strategy}")

    # Feature engineering
    engineered_features = []
    if engineer_features:
        print("Engineering features")
        df["hardness_solids_ratio"] = df["Hardness"] / (df["Solids"] + 1)
        engineered_features.append("hardness_solids_ratio")

    # Cap outliers
    def cap_outliers_iqr(column, threshold=1.5):
        """Cap outliers using the IQR method."""
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return column.clip(lower=lower_bound, upper=upper_bound)
    
    def cap_outliers_zscore(column, threshold=3.0):
        """Cap outliers using the z-score method."""
        mean = column.mean()
        std = column.std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        return column.clip(lower=lower_bound, upper=upper_bound)

    # Determine which columns to process
    if outlier_columns is None:
        outlier_columns = ["Sulfate", "ph", "Hardness"]
        if engineer_features:
            outlier_columns.append("hardness_solids_ratio")
    
    # Apply outlier capping
    if outlier_method and outlier_columns:
        print(f"Capping outliers using {outlier_method} method (threshold={outlier_threshold})")
        for col in outlier_columns:
            if col in df.columns:
                if outlier_method == "IQR":
                    df[col] = cap_outliers_iqr(df[col], threshold=outlier_threshold)
                elif outlier_method == "zscore":
                    df[col] = cap_outliers_zscore(df[col], threshold=outlier_threshold)
                else:
                    raise ValueError(f"Unknown outlier method: {outlier_method}")

    return df, engineered_features


def split_and_scale_data(df, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features.

    Args:
        df (pd.DataFrame): Processed dataframe with Potability target column
        test_size (float): Proportion of dataset for test set (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    print("\n")
    print("Splitting data...")
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_and_log(
    X_train, y_train, X_test, y_test, scaler,
    dataset_path="water_potability.csv", 
    experiment_name="water_potability",
    test_size=0.2, 
    random_state=42,
    # Data Preprocessing Parameters
    engineered_features=None, 
    outlier_method="IQR", 
    outlier_threshold=1.5, 
    missing_value_strategy="median",
    # Model Hyperparameters
    var_smoothing=1e-9,
    # Metadata
    run_source="cli"
):
    """
    Train model and log params/metrics/artifacts to MLflow AND Elasticsearch.

    Args:
        run_source (str): Source of the run - "cli", "api", "notebook", etc.

    Returns: trained model, run_id
    """
    if engineered_features is None:
        engineered_features = []
        
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        # Set tags for organization and filtering
        mlflow.set_tag("source", run_source)
        mlflow.set_tag("model_type", "GaussianNB")
        mlflow.set_tag("created_by", "train_and_log")

        # Log data parameters
        mlflow.log_param("data.dataset", dataset_path)
        mlflow.log_param("data.test_size", test_size)
        mlflow.log_param("data.random_state", random_state)
        mlflow.log_param("data.scaler", type(scaler).__name__)

        # Log preprocessing parameters
        mlflow.log_param("prep.missing_value_strategy", missing_value_strategy)
        mlflow.log_param("prep.engineer_features", len(engineered_features) > 0)
        mlflow.log_param("prep.engineered_features", ",".join(engineered_features) if engineered_features else "none")
        mlflow.log_param("prep.outlier_method", outlier_method if outlier_method else "none")
        mlflow.log_param("prep.outlier_threshold", outlier_threshold)
        
        # Log model hyperparameters
        mlflow.log_param("model.algorithm", "GaussianNB")
        mlflow.log_param("model.var_smoothing", var_smoothing)
        mlflow.log_param("timestamp", int(time.time()))

        # Train
        print("Creating model")
        model = GaussianNB(var_smoothing=var_smoothing)
        print("Training Model")
        model.fit(X_train, y_train)

        # Predict and log metrics
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        # ============================================
        # SEND TO ELASTICSEARCH
        # ============================================
        metrics_dict = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        }
        
        params_dict = {
            "dataset": dataset_path,
            "test_size": test_size,
            "random_state": random_state,
            "scaler": type(scaler).__name__,
            "missing_value_strategy": missing_value_strategy,
            "engineer_features": len(engineered_features) > 0,
            "engineered_features": ",".join(engineered_features) if engineered_features else "none",
            "outlier_method": outlier_method if outlier_method else "none",
            "outlier_threshold": outlier_threshold,
            "algorithm": "GaussianNB",
            "var_smoothing": var_smoothing
        }
        
        tags_dict = {
            "source": run_source,
            "model_type": "GaussianNB",
            "experiment": experiment_name
        }
        
        log_to_elasticsearch(
            run_id=run_id,
            metrics=metrics_dict,
            params=params_dict,
            tags=tags_dict,
            model_name="water_potability"
        )
        # ============================================

        # Log confusion matrix as image
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, display_labels=["Not Potable","Potable"])
        fig.tight_layout()
        cm_path = f"confusion_matrix_{run_id}.png"
        fig.savefig(cm_path)
        plt.close(fig)
        mlflow.log_artifact(cm_path)
        try:
            os.remove(cm_path)
        except Exception:
            pass

        # Log model as MLflow model
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model",
            input_example=X_train[:1],
        )
        
        # Register to model registry
        model_uri = f"runs:/{run_id}/model"
        registered_name = "water_potability_model"
        try:
            result = mlflow.register_model(model_uri, registered_name)
            client = mlflow.tracking.MlflowClient()
            # Add version tags
            client.set_model_version_tag(
                name=registered_name,
                version=result.version,
                key="source",
                value=run_source
            )
            print(f"Registered model: {result.name}, version: {result.version}")
        except Exception as e:
            print("Model registration failed or registry not available:", e)

        # Keep joblib artifacts too
        joblib_model_path = f"naive_bayes_{run_id}.pkl"
        joblib.dump(model, joblib_model_path)
        mlflow.log_artifact(joblib_model_path, artifact_path="artifacts")
        try:
            os.remove(joblib_model_path)
        except Exception:
            pass

        # Log scaler
        scaler_path = f"scaler_{run_id}.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="artifacts")
        try:
            os.remove(scaler_path)
        except Exception:
            pass

        meta = {"run_id": run_id, "accuracy": acc}
        meta_path = f"meta_{run_id}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        mlflow.log_artifact(meta_path, artifact_path="artifacts")
        try:
            os.remove(meta_path)
        except Exception:
            pass

    return model, run_id


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.

    Args:
        model (GaussianNB): Trained model
        X_test (np.ndarray): Scaled test features
        y_test (pd.Series): True test labels

    Returns:
        float: Model accuracy score
    """
    print("\n")
    y_pred = model.predict(X_test)

    print(type(X_test))

    accuracy = accuracy_score(y_test, y_pred)
    percesion = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {percesion:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(y_test, y_pred, target_names=["Not Potable", "Potable"])
    )

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nBreakdown:")
    print(f"True Negatives (correct non-potable):  {cm[0, 0]}")
    print(f"False Positives (wrong potable):       {cm[0, 1]}")
    print(f"False Negatives (wrong non-potable):   {cm[1, 0]}")
    print(f"True Positives (correct potable):      {cm[1, 1]}")

    return accuracy

def save_model(
    model,
    scaler,
    model_path="models/naive_bayes_model.pkl",
    scaler_path="models/scaler.pkl",
):
    """
    Save trained model and scaler to disk using joblib.

    Args:
        model (GaussianNB): Trained model to save
        scaler (StandardScaler): Fitted scaler to save
        model_path (str): Path to save model file
        scaler_path (str): Path to save scaler file

    Returns:
        None
    """
    print("\n")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    print("Saving model and scaler...")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")


def load_model(
    model_path="models/naive_bayes_model.pkl", scaler_path="models/scaler.pkl"
):
    """
    Load saved model and scaler from disk.

    Args:
        model_path (str): Path to saved model file
        scaler_path (str): Path to saved scaler file

    Returns:
        tuple: (model, scaler)
    """
    print("\n")
    print("Loading model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded!")
    return model, scaler
