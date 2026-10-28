from pathlib import Path

import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "artifacts"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# Reference data used during model development
reference_data = pd.read_csv(
    MODEL_DIR / "reference_data.csv"
)

# Demonstration current data.
# Replace this with real, approved UAT input data
# when it becomes available.
current_data = reference_data.sample(
    frac=1.0,
    random_state=7,
).copy()

report = Report(
    metrics=[
        DataDriftPreset(),
    ]
)

report.run(
    reference_data=reference_data,
    current_data=current_data,
)

report.save_html(
    str(REPORT_DIR / "data_drift_report.html")
)

print("Evidently monitoring report generated.")
print(
    "Open reports/data_drift_report.html "
    "in a browser to inspect the report."
)