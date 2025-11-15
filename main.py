"""
Main execution script for Water Potability Pipeline
Usage:
    python main.py --mode full
    python main.py --mode train --data water_potability.csv
    python main.py --mode evaluate
"""

import argparse
from model_pipeline import (
    prepare_data,
    split_and_scale_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

choices = ["prepare", "train", "evaluate", "full"]


def process(mode, data, model_path, scaler_path):
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

    Returns:
        None
    """
    if mode not in choices:
        return  # Exit if we get a bad mode

    if mode == "prepare":
        df = prepare_data(data)
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(df)
        print("✅ Data preparation complete!")

    elif mode == "train":
        df = prepare_data(data)
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(df)
        model = train_model(X_train, y_train)
        save_model(model, scaler, model_path, scaler_path)
        print("✅ Training complete!")

    elif mode == "evaluate":
        df = prepare_data(data)
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(df)
        model, loaded_scaler = load_model(model_path, scaler_path)
        evaluate_model(model, X_test, y_test)
        print("✅ Evaluation complete!")

    elif mode == "full":
        df = prepare_data(data)
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(df)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        save_model(model, scaler, model_path, scaler_path)
        print("✅ Full pipeline complete!")


def main():
    """
    Main function to parse arguments and execute the water potability pipeline.

    Configures argument parser for CLI interface and delegates execution
    to the process() function based on provided arguments.
    """
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Water Potability Classification Pipeline"
    )

    # Add arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=choices,
        help="Mode: train, evaluate, or full pipeline",
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

    # Parse arguments
    args = parser.parse_args()

    mode = args.mode
    data = args.data
    model_path = args.model_path
    scaler_path = args.scaler_path

    print("=" * 60)
    print("WATER POTABILITY CLASSIFICATION PIPELINE")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Dataset: {data}")
    print("=" * 60)

    process(mode, data, model_path, scaler_path)


if __name__ == "__main__":
    main()
