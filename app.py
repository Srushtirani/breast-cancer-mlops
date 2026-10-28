from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "artifacts"

app = FastAPI(
    title="Breast Cancer Classification API",
    description="Educational ML classification demonstration",
    version="1.0.0",
)

model = None
feature_names = None


def load_model():
    global model, feature_names

    model_path = MODEL_DIR / "model.joblib"
    features_path = MODEL_DIR / "feature_names.joblib"

    if model is None:
        if not model_path.exists() or not features_path.exists():
            raise RuntimeError(
                "Model files missing. Run python train.py first."
            )

        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        load_model()

        if len(request.features) != len(feature_names):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(feature_names)} features.",
            )

        if not all(
            pd.notna(value) and abs(value) != float("inf")
            for value in request.features
        ):
            raise HTTPException(
                status_code=400,
                detail="Features must contain finite numbers.",
            )

        input_data = pd.DataFrame(
            [request.features],
            columns=feature_names,
        )

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]

        label = "malignant" if prediction == 0 else "benign"

        return {
            "prediction": label,
            "class_id": prediction,
            "class_probabilities": {
                "malignant": float(probabilities[0]),
                "benign": float(probabilities[1]),
            },
            "notice": (
                "Educational output only; not a medical diagnosis."
            ),
        }

    except HTTPException:
        raise
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )