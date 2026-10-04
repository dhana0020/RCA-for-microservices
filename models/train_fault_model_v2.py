import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/rca_training_dataset.csv"
MODEL_PATH = "models/xgboost_fault_model_v2.pkl"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("FAULT MODEL V2")
print("=" * 70)

print(f"\nTotal rows: {len(df)}")


# ============================================================
# ROOT-CAUSE ROWS ONLY
# ============================================================

df = df[df["label"] == 1].copy()

print(f"Root-cause rows: {len(df)}")

print("\nFault distribution:")
print(df["fault"].value_counts())


# ============================================================
# FEATURES
# ============================================================

exclude_columns = [
    "service",
    "case",
    "dataset",
    "fault",
    "label"
]

feature_columns = [
    col for col in df.columns
    if col not in exclude_columns
    and pd.api.types.is_numeric_dtype(df[col])
]

X = df[feature_columns]
y = df["fault"]

print(f"\nNumber of features: {len(feature_columns)}")


# ============================================================
# ENCODE LABELS
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nClass mapping:")

for i, name in enumerate(label_encoder.classes_):
    print(f"{i} -> {name}")


# ============================================================
# GROUPED TRAIN / TEST SPLIT
# ============================================================

groups = df["case"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, y_encoded, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y_encoded[train_idx]
y_test = y_encoded[test_idx]


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# TRAIN XGBOOST
# ============================================================

model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

package = {
    "model": model,
    "feature_columns": feature_columns,
    "label_encoder": label_encoder
}

joblib.dump(
    package,
    MODEL_PATH
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(MODEL_PATH)