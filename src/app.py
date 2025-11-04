# src/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Optional
import os
import joblib
import numpy as np
import mlflow
import json
from datetime import datetime

app = FastAPI(title="Iris Predictor API")

TARGET_NAMES = ["setosa", "versicolor", "virginica"]

class PredictRequest(BaseModel):
    features: List[float]
    true: Optional[int] = None  # optional true label (0,1,2)

    @validator("features")
    def features_must_have_four(cls, v):
        if not isinstance(v, list):
            raise ValueError("features must be a list of four floats")
        if len(v) != 4:
            raise ValueError("features must contain exactly 4 numbers: [sepal_len, sepal_wid, petal_len, petal_wid]")
        return v

def find_model_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # primary location
    m1 = os.path.join(root, "models", "iris_model.pkl")
    if os.path.exists(m1):
        return m1
    # fallback: search mlruns artifacts
    mlruns = os.path.join(root, "mlruns")
    if os.path.isdir(mlruns):
        for exp in sorted(os.listdir(mlruns), reverse=True):
            exp_dir = os.path.join(mlruns, exp)
            if not os.path.isdir(exp_dir):
                continue
            for run in sorted(os.listdir(exp_dir), reverse=True):
                candidate = os.path.join(exp_dir, run, "artifacts", "saved_models", "iris_model.pkl")
                if os.path.exists(candidate):
                    return candidate
    return None

@app.on_event("startup")
def load_model():
    global MODEL, MODEL_PATH
    MODEL_PATH = find_model_path()
    if MODEL_PATH is None:
        raise RuntimeError("Model not found. Run training first to create models/iris_model.pkl or ensure mlruns artifacts exist.")
    MODEL = joblib.load(MODEL_PATH)

def mlflow_setup():
    # use sqlite mlflow DB in project root so UI sees requests
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(root, "mlflow.db")
    db_uri = f"sqlite:///{db_path.replace(os.sep, '/')}"
    mlflow.set_tracking_uri(db_uri)
    mlflow.set_experiment("Predictions")

@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}

@app.post("/predict")
def predict(req: PredictRequest):
    # prepare features
    features = np.array(req.features).reshape(1, -1)
    try:
        pred_id = int(MODEL.predict(features)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    pred_name = TARGET_NAMES[pred_id]

    # log to mlflow
    mlflow_setup()
    run_name = f"api-predict-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("input_features", json.dumps(req.features))
        mlflow.log_param("model_path", MODEL_PATH)
        mlflow.log_param("predicted_class_name", pred_name)
        mlflow.log_metric("predicted_class_id", float(pred_id))
        mlflow.log_param("timestamp", datetime.now().isoformat())
        if req.true is not None:
            mlflow.log_param("true_class_id", int(req.true))
            correct = 1.0 if int(req.true) == pred_id else 0.0
            mlflow.log_metric("correct", float(correct))

    return {"prediction_id": pred_id, "prediction_name": pred_name, "model_path": MODEL_PATH}
