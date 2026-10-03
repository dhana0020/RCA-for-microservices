
import pandas as pd
import numpy as np

from pathlib import Path

from xgboost import XGBClassifier

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import joblib


# ================================================================
# PATHS
# ================================================================

DATA_FILE = Path(
    "data/rca_training_dataset.csv"
)

MODEL_DIR = Path(
    "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    MODEL_DIR /
    "xgboost_rca_model.pkl"
)


# ================================================================
# LOAD DATASET
# ================================================================

print("\n========================================")
print("LOADING DATASET")
print("========================================")

df = pd.read_csv(
    DATA_FILE
)

print(
    "Dataset shape:",
    df.shape
)


# ================================================================
# IDENTIFY FEATURES
# ================================================================

# Columns that are NOT input features
NON_FEATURE_COLUMNS = [
    "service",
    "case",
    "dataset",
    "fault",
    "label"
]

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in NON_FEATURE_COLUMNS
]

print(
    "\nNumber of features:",
    len(FEATURE_COLUMNS)
)

print(
    "\nFeatures:"
)

for feature in FEATURE_COLUMNS:
    print(
        " -",
        feature
    )


# ================================================================
# INPUT / TARGET
# ================================================================

X = df[
    FEATURE_COLUMNS
].copy()

y = df[
    "label"
].astype(int)


# ================================================================
# HANDLE INVALID VALUES
# ================================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# ================================================================
# GROUP-BASED TRAIN / TEST SPLIT
# ================================================================
#
# IMPORTANT:
# All five service rows belonging to the same incident
# must stay in the same split.
#
# Otherwise the same incident could appear in both
# training and testing, causing data leakage.
# ================================================================

groups = df[
    "case"
]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X.iloc[
    train_indices
].copy()

X_test = X.iloc[
    test_indices
].copy()

y_train = y.iloc[
    train_indices
].copy()

y_test = y.iloc[
    test_indices
].copy()


print(
    "\n========================================"
)

print(
    "TRAIN / TEST SPLIT"
)

print(
    "========================================"
)

print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)

print(
    "Training cases:",
    df.iloc[
        train_indices
    ]["case"].nunique()
)

print(
    "Testing cases:",
    df.iloc[
        test_indices
    ]["case"].nunique()
)


# ================================================================
# TRAIN XGBOOST
# ================================================================

print(
    "\n========================================"
)

print(
    "TRAINING XGBOOST"
)

print(
    "========================================"
)


model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# ================================================================
# SERVICE-LEVEL BINARY EVALUATION
# ================================================================

y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print(
    "\n========================================"
)

print(
    "BINARY CLASSIFICATION RESULTS"
)

print(
    "========================================"
)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ================================================================
# ROOT-CAUSE RANKING
# ================================================================
#
# The actual application task is:
#
# For each incident:
#   1. Score every candidate service.
#   2. Rank services by probability.
#   3. Select the highest-probability service.
#
# Therefore ranking accuracy is more meaningful than
# ordinary binary accuracy.
# ================================================================

test_df = df.iloc[
    test_indices
].copy()

test_df = test_df.reset_index(
    drop=True
)

X_test_reset = X_test.reset_index(
    drop=True
)


# Predict probability that each service is the root cause

probabilities = model.predict_proba(
    X_test_reset
)[:, 1]


test_df[
    "root_cause_probability"
] = probabilities


# ================================================================
# CALCULATE TOP-1 ROOT CAUSE ACCURACY
# ================================================================

correct = 0
total_cases = 0

ranking_results = []


for case_name, case_group in test_df.groupby(
    "case"
):

    # Sort services from highest to lowest probability

    ranked = case_group.sort_values(
        "root_cause_probability",
        ascending=False
    )

    predicted_service = (
        ranked.iloc[0]["service"]
    )

    actual_service = (
        ranked.loc[
            ranked["label"] == 1,
            "service"
        ].iloc[0]
    )

    is_correct = (
        predicted_service
        == actual_service
    )

    if is_correct:
        correct += 1

    total_cases += 1

    ranking_results.append(
        {
            "case": case_name,
            "actual_root_cause": actual_service,
            "predicted_root_cause": predicted_service,
            "confidence": ranked.iloc[0][
                "root_cause_probability"
            ],
            "correct": is_correct
        }
    )


top1_accuracy = (
    correct / total_cases
    if total_cases > 0
    else 0.0
)


# ================================================================
# DISPLAY ROOT-CAUSE RESULTS
# ================================================================

ranking_results_df = pd.DataFrame(
    ranking_results
)


print(
    "\n========================================"
)

print(
    "ROOT-CAUSE SERVICE RESULTS"
)

print(
    "========================================"
)

print(
    "Test incidents:",
    total_cases
)

print(
    f"Top-1 RCA Accuracy: "
    f"{top1_accuracy:.4f}"
)

print(
    f"Top-1 RCA Accuracy: "
    f"{top1_accuracy * 100:.2f}%"
)


# ================================================================
# SAVE RANKING RESULTS
# ================================================================

ranking_file = Path(
    "data/rca_test_results.csv"
)

ranking_results_df.to_csv(
    ranking_file,
    index=False
)


print(
    "\nTest results saved to:"
)

print(
    ranking_file
)


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

importance = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_
    }
)

importance = importance.sort_values(
    "importance",
    ascending=False
)


print(
    "\n========================================"
)

print(
    "TOP FEATURE IMPORTANCE"
)

print(
    "========================================"
)

print(
    importance.head(15).to_string(
        index=False
    )
)


# ================================================================
# SAVE MODEL
# ================================================================

joblib.dump(
    {
        "model": model,
        "features": FEATURE_COLUMNS
    },
    MODEL_FILE
)


print(
    "\n========================================"
)

print(
    "MODEL SAVED"
)

print(
    "========================================"
)

print(
    MODEL_FILE
)

print(
    "\nTraining completed successfully!"
)

