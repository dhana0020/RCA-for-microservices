import sys
from pathlib import Path

import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_DIR = PROJECT_ROOT / "ml" / "features"
sys.path.insert(0, str(FEATURES_DIR))

from extract_features import extract_case_features
from service_config import SERVICE_CANDIDATES


API_URL = "http://127.0.0.1:8000/predict"

CASES_FILE = PROJECT_ROOT / "data" / "cases.parquet"
CASES_DIR = PROJECT_ROOT / "data" / "re2_cases"

# Start with 20 cases for testing
MAX_CASES = 270


print("Loading RCAEval cases...")

cases = pd.read_parquet(CASES_FILE)

# Only RE2 cases
cases = cases[cases["suite"] == "RE2"].copy()

# Take first MAX_CASES cases
cases = cases.head(MAX_CASES)

print("Cases to evaluate:", len(cases))


results = []

correct = 0
total = 0


for index, case_row in cases.iterrows():

    case_name = case_row["case"]
    dataset = case_row["dataset"]

    actual_root_cause = case_row["root_cause_service"]
    actual_fault = case_row["fault"]

    print("\n" + "=" * 60)
    print("Case:", case_name)
    print("Expected:", actual_root_cause, "|", actual_fault)

    case_path = CASES_DIR / case_name

    try:

        # -----------------------------------------
        # Extract features
        # -----------------------------------------

        features = extract_case_features(case_path)

        valid_services = SERVICE_CANDIDATES[dataset]

        features = features[
            features["service"].isin(valid_services)
        ].copy()

        if features.empty:
            print("No valid services found. Skipping.")
            continue

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

            services.append(service_data)

        payload = {
            "application": dataset,
            "services": services
        }

        # -----------------------------------------
        # Send to FastAPI
        # -----------------------------------------

        response = requests.post(
            API_URL,
            json=payload
        )

        if response.status_code != 200:

            print("API Error:", response.text)

            results.append({
                "case": case_name,
                "application": dataset,
                "actual_root_cause": actual_root_cause,
                "predicted_root_cause": "API_ERROR",
                "actual_fault": actual_fault,
                "predicted_fault": "API_ERROR",
                "root_cause_confidence": 0,
                "fault_confidence": 0,
                "correct": False
            })

            continue

        prediction = response.json()

        predicted_root_cause = prediction["root_cause_service"]
        predicted_fault = prediction["fault_type"]

        root_confidence = prediction["root_cause_confidence"]
        fault_confidence = prediction["fault_confidence"]

        is_correct = (
            predicted_root_cause == actual_root_cause
        )

        if is_correct:
            correct += 1

        total += 1

        print(
            "Predicted:",
            predicted_root_cause,
            "|",
            predicted_fault
        )

        print(
            "RCA confidence:",
            root_confidence
        )

        print(
            "Fault confidence:",
            fault_confidence
        )

        print(
            "RCA:",
            "CORRECT" if is_correct else "WRONG"
        )

        results.append({
            "case": case_name,
            "application": dataset,
            "actual_root_cause": actual_root_cause,
            "predicted_root_cause": predicted_root_cause,
            "actual_fault": actual_fault,
            "predicted_fault": predicted_fault,
            "root_cause_confidence": root_confidence,
            "fault_confidence": fault_confidence,
            "correct": is_correct
        })

    except Exception as e:

        print("Error:", e)

        results.append({
            "case": case_name,
            "application": dataset,
            "actual_root_cause": actual_root_cause,
            "predicted_root_cause": "ERROR",
            "actual_fault": actual_fault,
            "predicted_fault": "ERROR",
            "root_cause_confidence": 0,
            "fault_confidence": 0,
            "correct": False
        })


# -----------------------------------------
# Final evaluation
# -----------------------------------------

results_df = pd.DataFrame(results)

output_path = PROJECT_ROOT / "data" / "api_evaluation_results.csv"

results_df.to_csv(
    output_path,
    index=False
)


print("\n")
print("=" * 60)
print("FINAL API EVALUATION")
print("=" * 60)

print("Total evaluated:", total)
print("Correct:", correct)

if total > 0:

    accuracy = correct / total

    print(
        "Top-1 RCA Accuracy:",
        f"{accuracy * 100:.2f}%"
    )

print("\nResults saved to:")
print(output_path)