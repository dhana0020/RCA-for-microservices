import pandas as pd

RESULTS_PATH = "data/api_evaluation_results.csv"

df = pd.read_csv(RESULTS_PATH)

print("=" * 70)
print("API EVALUATION SUMMARY")
print("=" * 70)

print(f"Total cases : {len(df)}")

print("\nColumns:")
print(df.columns.tolist())

# Find the actual column names used in your CSV
actual_col = "actual_root_cause"
predicted_col = "predicted_root_cause"

if actual_col not in df.columns or predicted_col not in df.columns:
    print("\nERROR: Expected columns were not found.")
    print("Check the column names printed above.")
    exit()

df["correct"] = df[actual_col] == df[predicted_col]

correct = df["correct"].sum()
incorrect = len(df) - correct

print(f"\nCorrect   : {correct}")
print(f"Incorrect : {incorrect}")
print(f"Accuracy  : {correct / len(df) * 100:.2f}%")

# Show failed cases
failed = df[~df["correct"]]

print("\n" + "=" * 70)
print("FAILED CASES")
print("=" * 70)

if len(failed) == 0:
    print("No failed cases!")
else:
    for _, row in failed.iterrows():

        print("\n" + "-" * 70)

        print("Case ID              :", row.get("case_id", "N/A"))
        print("Application          :", row.get("application", "N/A"))
        print("Actual root cause    :", row[actual_col])
        print("Predicted root cause :", row[predicted_col])

        if "actual_fault_type" in row:
            print("Actual fault type    :", row["actual_fault_type"])

        if "predicted_fault_type" in row:
            print("Predicted fault type :", row["predicted_fault_type"])

        if "root_cause_confidence" in row:
            print("Root cause confidence:", row["root_cause_confidence"])

# Save failed cases
failed.to_csv(
    "data/failed_api_cases.csv",
    index=False
)

print("\n" + "=" * 70)
print("Failed cases saved to:")
print("data/failed_api_cases.csv")
print("=" * 70)