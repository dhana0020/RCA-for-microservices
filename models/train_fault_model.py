import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_FILE = "data/rca_training_dataset.csv"

df = pd.read_csv(DATA_FILE)

print("Original dataset shape:", df.shape)


# ============================================================
# 2. KEEP ONLY ROOT-CAUSE SERVICE ROWS
# ============================================================

# Each incident has 5 candidate services.
# label = 1 means that service is the actual root cause.

df = df[df["label"] == 1].copy()

print("Root-cause rows:", len(df))


# ============================================================
# 3. SELECT FEATURES
# ============================================================

DROP_COLUMNS = [
    "service",
    "case",
    "dataset",
    "fault",
    "label"
]

FEATURE_COLUMNS = [
    col for col in df.columns
    if col not in DROP_COLUMNS
]

X = df[FEATURE_COLUMNS].copy()
y = df["fault"].copy()
groups = df["case"]


# Handle missing / infinite values
X = X.replace([float("inf"), float("-inf")], 0)
X = X.fillna(0)


print("Number of features:", len(FEATURE_COLUMNS))
print("Fault classes:")
print(y.value_counts())


# ============================================================
# 4. ENCODE FAULT TYPES
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nFault class mapping:")

for number, fault_name in enumerate(label_encoder.classes_):
    print(number, "->", fault_name)


# ============================================================
# 5. GROUPED TRAIN / TEST SPLIT
# ============================================================

# Important:
# All rows belonging to the same incident must stay
# in either training OR testing, never both.

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y_encoded, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y_encoded[train_idx]
y_test = y_encoded[test_idx]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print(
    "Training cases:",
    df.iloc[train_idx]["case"].nunique()
)

print(
    "Testing cases:",
    df.iloc[test_idx]["case"].nunique()
)


# ============================================================
# 6. TRAIN XGBOOST FAULT MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,

    objective="multi:softprob",
    num_class=len(label_encoder.classes_),

    eval_metric="mlogloss",

    random_state=42,
    n_jobs=-1
)

print("\nTraining fault type model...")

model.fit(
    X_train,
    y_train
)


# ============================================================
# 7. PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("\n")
print("=" * 60)
print("FAULT TYPE CLASSIFICATION RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# ============================================================
# 9. SAVE TEST RESULTS
# ============================================================

results = df.iloc[test_idx][
    ["case", "dataset", "service", "fault"]
].copy()

results["predicted_fault"] = label_encoder.inverse_transform(
    y_pred
)

results.to_csv(
    "data/fault_test_results.csv",
    index=False
)

print(
    "\nTest results saved to:"
    "\ndata/fault_test_results.csv"
)


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\n")
print("=" * 60)
print("TOP FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance.head(15).to_string(index=False)
)


# ============================================================
# 11. SAVE MODEL
# ============================================================

model_package = {
    "model": model,
    "feature_columns": FEATURE_COLUMNS,
    "label_encoder": label_encoder
}

joblib.dump(
    model_package,
    "models/xgboost_fault_model.pkl"
)

print("\n")
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    "models/xgboost_fault_model.pkl"
)

print("\nFault type training completed successfully!")