# src/predict.py
import os
import argparse
import joblib
import json
import numpy as np
import mlflow
from datetime import datetime

TARGET_NAMES = ["setosa", "versicolor", "virginica"]

def find_model_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # prefer models/iris_model.pkl
    m1 = os.path.join(root, "models", "iris_model.pkl")
    if os.path.exists(m1):
        return m1

    # fallback: search mlruns for a saved artifact (latest first)
    mlruns = os.path.join(root, "mlruns")
    if os.path.isdir(mlruns):
        for exp in sorted(os.listdir(mlruns), reverse=True):
            exp_dir = os.path.join(mlruns, exp)
            if not os.path.isdir(exp_dir):
                continue
            for run in sorted(os.listdir(exp_dir), reverse=True):
                candidate = os.path.join(
                    exp_dir, run, "artifacts", "saved_models", "iris_model.pkl"
                )
                if os.path.exists(candidate):
                    return candidate
    return None

def load_model():
    path = find_model_path()
    if not path:
        raise FileNotFoundError("Could not find iris_model.pkl (looked in models/ and mlruns/)")
    return joblib.load(path), path

def predict(model, features):
    arr = np.array(features).reshape(1, -1)
    pred = int(model.predict(arr)[0])
    return pred

def mlflow_setup():
    # point MLflow to the same sqlite DB used by the UI in project root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(root, "mlflow.db")
    # sqlite URI requires forward slashes
    db_uri = f"sqlite:///{db_path.replace(os.sep, '/')}"
    mlflow.set_tracking_uri(db_uri)
    mlflow.set_experiment("Predictions")
    return db_uri

def log_prediction_to_mlflow(features, pred_id, pred_name, model_path, true_id=None):
    # ensure MLflow is configured
    mlflow_db = mlflow_setup()
    run_name = f"predict-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        # store features as JSON param for readability (small inputs OK)
        mlflow.log_param("input_features", json.dumps(features))
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("predicted_class_name", pred_name)
        mlflow.log_metric("predicted_class_id", float(pred_id))
        mlflow.log_param("timestamp", datetime.now().isoformat())

        # optional true label logging
        if true_id is not None:
            mlflow.log_param("true_class_id", int(true_id))
            correct = 1.0 if int(true_id) == int(pred_id) else 0.0
            mlflow.log_metric("correct", float(correct))

def parse_args():
    parser = argparse.ArgumentParser(description="Predict Iris species and log to MLflow.")
    parser.add_argument("features", nargs="*", type=float, help="Four numeric features (sepal_len sepal_wid petal_len petal_wid)")
    parser.add_argument("--csv", type=str, help='Comma-separated features e.g. "5.1,3.5,1.4,0.2"')
    parser.add_argument("--true", type=int, choices=[0,1,2], help="(Optional) true class id to compute correctness (0=setosa,1=versicolor,2=virginica)")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.csv:
        features = [float(x) for x in args.csv.split(",")]
    elif args.features and len(args.features) == 4:
        features = list(args.features)
    else:
        print("Usage:")
        print("  python src/predict.py 5.1 3.5 1.4 0.2")
        print('Or:')
        print('  python src/predict.py --csv "5.1,3.5,1.4,0.2"')
        print("Optional: add --true <0|1|2> to log true label and correctness.")
        return

    model, path = load_model()
    pred_id = predict(model, features)
    pred_name = TARGET_NAMES[pred_id]

    print(f"Loaded model from: {path}")
    print(f"Input: {features}")
    print(f"Predicted class id: {pred_id}")
    print(f"Predicted class name: {pred_name}")

    # Log to MLflow (will use sqlite DB in project root so UI shows it)
    log_prediction_to_mlflow(features, pred_id, pred_name, path, true_id=args.true)
    print("\n✅ Prediction logged to MLflow (experiment: 'Predictions').")
    if args.true is not None:
        print(f"True class id: {args.true} -> logged 'correct' metric accordingly.")

if __name__ == "__main__":
    main()
