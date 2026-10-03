import sys
from pathlib import Path

import requests
import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add ml/features to Python path
FEATURES_DIR = PROJECT_ROOT / "ml" / "features"

sys.path.insert(0, str(FEATURES_DIR))

from extract_features import extract_case_features
from service_config import SERVICE_CANDIDATES











# ============================================================
# 1. CASE TO TEST
# ============================================================

CASE_NAME = "re2ob_checkoutservice_cpu_1"

CASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "re2_cases"
    / CASE_NAME
)

APPLICATION = "RE2-OB"

API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# 2. EXTRACT REAL FEATURES
# ============================================================

print("Extracting real RCAEval features...")

features = extract_case_features(
    CASE_PATH
)


# ============================================================
# 3. KEEP ONLY VALID SERVICES
# ============================================================

valid_services = SERVICE_CANDIDATES[
    APPLICATION
]

features = features[
    features["service"].isin(valid_services)
].copy()


print("\nServices found:")

print(
    features["service"].tolist()
)


# ============================================================
# 4. CONVERT FEATURES TO API FORMAT
# ============================================================

services = []

for _, row in features.iterrows():

    service_data = {
        "service": row["service"]
    }

    for column in features.columns:

        if column == "service":
            continue

        value = row[column]

        if pd.isna(value):
            value = 0.0

        if value == float("inf") or value == float("-inf"):
            value = 0.0

        service_data[column] = float(value)

    services.append(
        service_data
    )


# ============================================================
# 5. CREATE REQUEST
# ============================================================

payload = {

    "application": APPLICATION,

    "services": services
}


print("\nSending prediction request to FastAPI...")


# ============================================================
# 6. CALL FASTAPI
# ============================================================

response = requests.post(
    API_URL,
    json=payload
)


# ============================================================
# 7. DISPLAY RESULT
# ============================================================

print("\nHTTP Status:", response.status_code)

print("\nAPI Response:")

print(
    response.json()
)


# ============================================================
# 8. EXPECTED RESULT
# ============================================================

print("\n")
print("=" * 60)

print("EXPECTED ROOT CAUSE:")
print("checkoutservice")

print("=" * 60)
