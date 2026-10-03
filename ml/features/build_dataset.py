
import pandas as pd
from pathlib import Path

from extract_features import extract_case_features
from service_config import SERVICE_CANDIDATES


# ================================================================
# PATHS
# ================================================================

CASES_FILE = "data/cases.parquet"
CASES_DIR = Path("data/re2_cases")

OUTPUT_FILE = Path(
    "data/rca_training_dataset.csv"
)


# ================================================================
# LOAD CASE INDEX
# ================================================================

cases = pd.read_parquet(
    CASES_FILE
)


# ================================================================
# KEEP ONLY RE2 CASES
# ================================================================

cases = cases[
    cases["dataset"].isin(
        [
            "RE2-OB",
            "RE2-SS",
            "RE2-TT"
        ]
    )
].reset_index(drop=True)


print(
    "Total RE2 cases:",
    len(cases)
)


# ================================================================
# BUILD FEATURE DATASET
# ================================================================

all_rows = []


for index, case in cases.iterrows():

    dataset_name = case["dataset"]
    case_name = case["case"]
    root_cause = case["root_cause_service"]

    case_path = (
        CASES_DIR / case_name
    )

    print(
        f"[{index + 1}/{len(cases)}] "
        f"{case_name}"
    )

    # ------------------------------------------------------------
    # EXTRACT FEATURES
    # ------------------------------------------------------------

    features = extract_case_features(
        case_path
    )

    # ------------------------------------------------------------
    # KEEP ONLY VALID ROOT-CAUSE CANDIDATES
    # ------------------------------------------------------------

    valid_services = (
        SERVICE_CANDIDATES[
            dataset_name
        ]
    )

    features = features[
        features["service"].isin(
            valid_services
        )
    ].copy()

    # ------------------------------------------------------------
    # ADD CASE INFORMATION
    # ------------------------------------------------------------

    features["case"] = case_name

    features["dataset"] = dataset_name

    features["fault"] = case["fault"]

    # ------------------------------------------------------------
    # CREATE BINARY LABEL
    #
    # Root-cause service = 1
    # Other candidate services = 0
    # ------------------------------------------------------------

    features["label"] = (
        features["service"]
        == root_cause
    ).astype(int)

    # ------------------------------------------------------------
    # STORE CASE FEATURES
    # ------------------------------------------------------------

    all_rows.append(
        features
    )


# ================================================================
# COMBINE ALL CASES
# ================================================================

dataset = pd.concat(
    all_rows,
    ignore_index=True
)


# ================================================================
# SAVE DATASET
# ================================================================

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)


# ================================================================
# DISPLAY RESULTS
# ================================================================

print("\n========================================")
print("DATASET CREATED")
print("========================================")

print(
    "Rows:",
    len(dataset)
)

print(
    "Columns:",
    len(dataset.columns)
)

print(
    "\nLabel distribution:"
)

print(
    dataset["label"].value_counts()
)

print(
    "\nCases:"
)

print(
    dataset["case"].nunique()
)

print(
    "\nServices:"
)

print(
    dataset["service"].nunique()
)

print(
    "\nApplications:"
)

print(
    dataset["dataset"].value_counts()
)

print(
    "\nFault types:"
)

print(
    dataset["fault"].value_counts()
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)

