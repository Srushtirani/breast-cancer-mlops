from pathlib import Path

import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "artifacts"

MODEL_DIR.mkdir(exist_ok=True)

# Load dataset
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target

# Save a reference dataset for monitoring
X.to_csv(MODEL_DIR / "reference_data.csv", index=False)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Build preprocessing + classification pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000)),
])

# Train model
model.fit(X_train, y_train)

# Evaluate on held-out test data
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("Classification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=dataset.target_names,
        zero_division=0,
    )
)

# Save trained model and feature names
joblib.dump(model, MODEL_DIR / "model.joblib")

joblib.dump(
    list(X.columns),
    MODEL_DIR / "feature_names.joblib",
)

print("Model training completed successfully.")