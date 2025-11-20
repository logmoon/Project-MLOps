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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


def prepare_data(dataset_path):
    """
    Load and preprocess the water potability dataset.

    Performs the following steps:
    1. Loads CSV data
    2. Fills missing values with column medians
    3. Engineers hardness_solids_ratio feature
    4. Caps outliers for Sulfate, pH, Hardness, and hardness_solids_ratio

    Args:
        dataset_path (str): Path to the water potability CSV file

    Returns:
        pd.DataFrame: Processed dataframe with engineered features and capped outliers
    """
    print("\n")  # Add some space
    print("Reading dataset")
    df = pd.read_csv(dataset_path)
    print("Processing dataset")
    # Fill null with medians (seems to have done the best job)
    df.fillna(df.median(), inplace=True)

    # Features engineering, this one seems to have given the best results
    df["hardness_solids_ratio"] = df["Hardness"] / (
        df["Solids"] + 1
    )  # +1 to avoid division by zero

    # Caps outliers, simple
    def cap_outliers(column):
        """
        Cap outliers using the IQR method.

        Args:
            column (pd.Series): Column to cap outliers

        Returns:
            pd.Series: Column with outliers capped at 1.5*IQR bounds
        """
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return column.clip(lower=lower_bound, upper=upper_bound)

    # Apply to problematic features
    df["Sulfate"] = cap_outliers(df["Sulfate"])
    df["ph"] = cap_outliers(df["ph"])
    df["Hardness"] = cap_outliers(df["Hardness"])
    df["hardness_solids_ratio"] = cap_outliers(df["hardness_solids_ratio"])

    return df


def split_and_scale_data(df, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features.

    Args:
        df (pd.DataFrame): Processed dataframe with Potability target column
        test_size (float): Proportion of dataset for test set (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
            - X_train_scaled (np.ndarray): Scaled training features
            - X_test_scaled (np.ndarray): Scaled test features
            - y_train (pd.Series): Training labels
            - y_test (pd.Series): Test labels
            - scaler (StandardScaler): Fitted scaler object
    """
    print("\n")  # Add some space
    print("Splitting data...")
    # Separate features and target
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train):
    """
    Train a Gaussian Naive Bayes classifier.

    Args:
        X_train (np.ndarray): Scaled training features
        y_train (pd.Series): Training labels (0: Not Potable, 1: Potable)

    Returns:
        GaussianNB: Trained Naive Bayes model
    """
    print("\n")  # Add some space
    print("Creating model")
    model = GaussianNB()
    print("Training Model")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.

    Prints:
    - Overall accuracy
    - Classification report (precision, recall, f1-score per class)
    - Confusion matrix with breakdown

    Args:
        model (GaussianNB): Trained model
        X_test (np.ndarray): Scaled test features
        y_test (pd.Series): True test labels

    Returns:
        float: Model accuracy score
    """
    print("\n")  # Add some space
    # Predict on test set
    y_pred = model.predict(X_test)

    print(type(X_test))

    # Calculate accuracy on ALL test data
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Detailed metrics
    print("\nClassification Report:")
    print(
        classification_report(y_test, y_pred, target_names=["Not Potable", "Potable"])
    )

    # Confusion matrix
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

    Creates necessary directories if they don't exist.

    Args:
        model (GaussianNB): Trained model to save
        scaler (StandardScaler): Fitted scaler to save
        model_path (str): Path to save model file (default: 'models/naive_bayes_model.pkl')
        scaler_path (str): Path to save scaler file (default: 'models/scaler.pkl')

    Returns:
        None
    """
    print("\n")  # Add some space
    # Make sure directories exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    # Save
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
        model_path (str): Path to saved model file (default: 'models/naive_bayes_model.pkl')
        scaler_path (str): Path to saved scaler file (default: 'models/scaler.pkl')

    Returns:
        tuple: (model, scaler)
            - model (GaussianNB): Loaded trained model
            - scaler (StandardScaler): Loaded fitted scaler
    """
    print("\n")  # Add some space
    print("Loading model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded!")
    return model, scaler
