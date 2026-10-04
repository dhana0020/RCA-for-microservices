import pandas as pd

DATA_PATH = "data/rca_training_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("FAULT FEATURE ANALYSIS")
print("=" * 70)

print(f"\nTotal rows: {len(df)}")

# --------------------------------------------------
# Check required columns
# --------------------------------------------------

required_columns = ["fault", "label"]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    print("\nMissing columns:", missing)
    print("\nAvailable columns:")
    print(df.columns.tolist())
    exit()

# --------------------------------------------------
# Root-cause rows only
# --------------------------------------------------

root_df = df[df["label"] == 1].copy()

print(f"Root-cause rows: {len(root_df)}")

print("\nFault distribution:")
print(root_df["fault"].value_counts())

# --------------------------------------------------
# Feature columns
# --------------------------------------------------

exclude = [
    "case",
    "application",
    "service",
    "fault",
    "label"
]

feature_columns = [
    col for col in df.columns
    if col not in exclude
    and pd.api.types.is_numeric_dtype(df[col])
]

print("\nNumber of numeric features:", len(feature_columns))

# --------------------------------------------------
# Compare DELAY vs LOSS
# --------------------------------------------------

delay_df = root_df[root_df["fault"] == "delay"]
loss_df = root_df[root_df["fault"] == "loss"]

print("\n" + "=" * 70)
print("DELAY vs LOSS")
print("=" * 70)

print(f"\nDelay samples: {len(delay_df)}")
print(f"Loss samples : {len(loss_df)}")

comparison = []

for feature in feature_columns:

    delay_mean = delay_df[feature].mean()
    loss_mean = loss_df[feature].mean()

    difference = abs(delay_mean - loss_mean)

    comparison.append({
        "feature": feature,
        "delay_mean": delay_mean,
        "loss_mean": loss_mean,
        "absolute_difference": difference
    })

comparison_df = pd.DataFrame(comparison)

comparison_df = comparison_df.sort_values(
    "absolute_difference",
    ascending=False
)

print("\nTop features distinguishing DELAY vs LOSS:")

print(
    comparison_df.head(15).to_string(index=False)
)

# --------------------------------------------------
# Feature statistics for all fault types
# --------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE MEANS BY FAULT TYPE")
print("=" * 70)

means = root_df.groupby("fault")[feature_columns].mean()

print(means.to_string())

# --------------------------------------------------
# Save analysis
# --------------------------------------------------

comparison_df.to_csv(
    "data/delay_vs_loss_feature_analysis.csv",
    index=False
)

means.to_csv(
    "data/fault_feature_means.csv"
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print("data/delay_vs_loss_feature_analysis.csv")
print("data/fault_feature_means.csv")

print("\nDONE")