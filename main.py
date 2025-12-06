"""
Main execution script for Water Potability Pipeline
Usage:
    python main.py --mode full
    python main.py --mode train --missing_strategy mean --outlier_method zscore
    python main.py --mode evaluate
    python main.py --mode train --no_feature_engineering --outlier_threshold 2.0
"""

import argparse
from model_pipeline import (
    prepare_data,
    split_and_scale_data,
    evaluate_model,
    train_and_log,
    save_model,
    load_model,
)

choices = ["prepare", "train", "evaluate", "full"]


def process(mode, data, model_path, scaler_path, 
            missing_strategy, engineer_features, outlier_method, outlier_threshold,
            test_size, random_state, var_smoothing):
    """
    Process the ML pipeline based on the selected mode.

    Executes pipeline steps based on the passed mode:
    - train: prepare + split + train + save
    - evaluate: load + evaluate
    - full: All steps

    Args:
        mode (str): Execution mode from choices list
        data (str): Path to dataset CSV file
        model_path (str): Path to save/load trained model
        scaler_path (str): Path to save/load fitted scaler
        missing_strategy (str): Strategy for handling missing values
        engineer_features (bool): Whether to engineer features
        outlier_method (str): Method for outlier detection
        outlier_threshold (float): Threshold for outlier capping
        test_size (float): Test set proportion
        random_state (int): Random seed
        var_smoothing (float): GaussianNB var_smoothing hyperparameter

    Returns:
        None
    """
    if mode not in choices:
        return

    if mode == "prepare":
        df, engineered_features = prepare_data(
            data,
            missing_value_strategy=missing_strategy,
            engineer_features=engineer_features,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold
        )
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(
            df, test_size=test_size, random_state=random_state
        )
        print("✅ Data preparation complete!")

    elif mode == "train":
        df, engineered_features = prepare_data(
            data,
            missing_value_strategy=missing_strategy,
            engineer_features=engineer_features,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold
        )
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(
            df, test_size=test_size, random_state=random_state
        )
        model, run_id = train_and_log(
            X_train, y_train, X_test, y_test, scaler,
            dataset_path=data,
            test_size=test_size,
            random_state=random_state,
            experiment_name="water_potability",
            engineered_features=engineered_features,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold,
            missing_value_strategy=missing_strategy,
            var_smoothing=var_smoothing
        )
        save_model(model, scaler, model_path, scaler_path)
        print(f"✅ Training complete! MLflow run: {run_id}")

    elif mode == "evaluate":
        df, engineered_features = prepare_data(
            data,
            missing_value_strategy=missing_strategy,
            engineer_features=engineer_features,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold
        )
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(
            df, test_size=test_size, random_state=random_state
        )
        model, loaded_scaler = load_model(model_path, scaler_path)
        evaluate_model(model, X_test, y_test)
        print("✅ Evaluation complete!")

    elif mode == "full":
        df, engineered_features = prepare_data(
            data,
            missing_value_strategy=missing_strategy,
            engineer_features=engineer_features,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold
        )
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(
            df, test_size=test_size, random_state=random_state
        )
        model, run_id = train_and_log(
            X_train, y_train, X_test, y_test, scaler,
            dataset_path=data,
            test_size=test_size,
            random_state=random_state,
            experiment_name="water_potability",
            engineered_features=engineered_features,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold,
            missing_value_strategy=missing_strategy,
            var_smoothing=var_smoothing
        )
        print(f"Training complete! MLflow run: {run_id}")
        evaluate_model(model, X_test, y_test)
        save_model(model, scaler, model_path, scaler_path)
        print("✅ Full pipeline complete!")


def main():
    """
    Main function to parse arguments and execute the water potability pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Water Potability Classification Pipeline"
    )

    # Mode and file paths
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=choices,
        help="Mode: prepare, train, evaluate, or full pipeline",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="water_potability.csv",
        help="Path to dataset CSV file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/naive_bayes_model.pkl",
        help="Path to save/load model",
    )
    parser.add_argument(
        "--scaler_path",
        type=str,
        default="models/scaler.pkl",
        help="Path to save/load scaler",
    )

    # Preprocessing parameters
    parser.add_argument(
        "--missing_strategy",
        type=str,
        default="median",
        choices=["median", "mean", "drop"],
        help="Strategy for handling missing values (default: median)",
    )
    parser.add_argument(
        "--no_feature_engineering",
        action="store_true",
        help="Disable feature engineering (default: enabled)",
    )
    parser.add_argument(
        "--outlier_method",
        type=str,
        default="IQR",
        choices=["IQR", "zscore", "none"],
        help="Method for outlier detection (default: IQR)",
    )
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=1.5,
        help="Threshold for outlier capping (default: 1.5 for IQR, use 3.0 for zscore)",
    )

    # Train/test split parameters
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of dataset for test set (default: 0.2)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Model hyperparameters
    parser.add_argument(
        "--var_smoothing",
        type=float,
        default=1e-9,
        help="GaussianNB var_smoothing parameter (default: 1e-9)",
    )

    args = parser.parse_args()

    # Convert flags
    engineer_features = not args.no_feature_engineering
    outlier_method = None if args.outlier_method == "none" else args.outlier_method

    print("=" * 60)
    print("WATER POTABILITY CLASSIFICATION PIPELINE")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.data}")
    print(f"\nPreprocessing Configuration:")
    print(f"  Missing Value Strategy: {args.missing_strategy}")
    print(f"  Feature Engineering: {engineer_features}")
    print(f"  Outlier Method: {outlier_method or 'none'}")
    print(f"  Outlier Threshold: {args.outlier_threshold}")
    print(f"\nModel Configuration:")
    print(f"  Test Size: {args.test_size}")
    print(f"  Random State: {args.random_state}")
    print(f"  Var Smoothing: {args.var_smoothing}")
    print("=" * 60)

    process(
        args.mode, args.data, args.model_path, args.scaler_path,
        args.missing_strategy, engineer_features, outlier_method, args.outlier_threshold,
        args.test_size, args.random_state, args.var_smoothing
    )


if __name__ == "__main__":
    main()