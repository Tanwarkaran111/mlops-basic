# src/train.py
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def main():
    # ensure output folder exists (relative to project root)
    os.makedirs("../models", exist_ok=True)

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    target_names = iris.target_names
    print("\nSample predictions:")
    for i in range(5):
        print(f"Predicted: {target_names[preds[i]]}, Actual: {target_names[y_test[i]]}")

    # ✅ Dynamic model path (so it always works no matter where you run)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root_dir, "models", "iris_model.pkl")

    joblib.dump(model, model_path)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()
