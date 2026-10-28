import numpy as np
import pandas as pd

from fastapi.testclient import TestClient

from app import app
from train import MODEL_DIR

client = TestClient(app)


def setup_module():
    # Ensure model artifacts exist before tests run.
    import train


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_artifacts_exist():
    assert (MODEL_DIR / "model.joblib").exists()
    assert (MODEL_DIR / "feature_names.joblib").exists()


def test_prediction_endpoint():
    from sklearn.datasets import load_breast_cancer

    dataset = load_breast_cancer(as_frame=True)
    sample = dataset.data.iloc[0].tolist()

    response = client.post(
        "/predict",
        json={"features": sample},
    )

    assert response.status_code == 200
    assert response.json()["prediction"] in [
        "benign",
        "malignant",
    ]


def test_invalid_feature_count():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0]},
    )

    assert response.status_code == 400


def test_non_finite_input():
    response = client.post(
        "/predict",
        json={"features": [float("nan")] * 30},
    )

    assert response.status_code == 422