# src/train.py
import os
import yaml
import mlflow
import mlflow.sklearn
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    # ---- determine project root & paths ----
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "iris_model.pkl")
    params_path = os.path.join(root_dir, "params.yaml")

    # ---- load params ----
    with open(params_path, "r") as f:
        params = yaml.safe_load(f).get("train", {})

    max_iter = params.get("max_iter", 200)
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)
    sample_print = params.get("sample_print", 5)

    # ---- load data ----
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # ---- split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ---- MLflow experiment ----
    mlflow.set_experiment("Iris-Classifier")

    with mlflow.start_run():
        # log params
        mlflow.log_params({
            "max_iter": max_iter,
            "test_size": test_size,
            "random_state": random_state
        })

        # train
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train, y_train)

        # predict + metric
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", float(acc))

        # show some predictions to console
        print(f"Accuracy: {acc:.4f}\n")
        print("Sample predictions:")
        for i in range(min(sample_print, len(preds))):
            print(f"Predicted: {target_names[preds[i]]}, Actual: {target_names[y_test[i]]}")

        # save artifact to models dir (and also log model artifact to MLflow)
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "sklearn-model")
        mlflow.log_artifact(model_path, artifact_path="saved_models")

        print(f"\nSaved model to {model_path}")

if __name__ == "__main__":
    main()
