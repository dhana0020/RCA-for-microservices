import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

RESULTS_PATH = "data/api_evaluation_results.csv"

df = pd.read_csv(RESULTS_PATH)

print("=" * 70)
print("COMPLETE API EVALUATION")
print("=" * 70)

total = len(df)

# --------------------------------------------------
# ROOT CAUSE
# --------------------------------------------------

root_correct = (
    df["actual_root_cause"] == df["predicted_root_cause"]
)

root_accuracy = root_correct.mean() * 100

print("\nROOT CAUSE SERVICE")
print("-" * 70)
print(f"Total cases        : {total}")
print(f"Correct            : {root_correct.sum()}")
print(f"Incorrect          : {total - root_correct.sum()}")
print(f"Top-1 Accuracy     : {root_accuracy:.2f}%")

# --------------------------------------------------
# FAULT TYPE
# --------------------------------------------------

fault_correct = (
    df["actual_fault"] == df["predicted_fault"]
)

fault_accuracy = fault_correct.mean() * 100

fault_f1 = f1_score(
    df["actual_fault"],
    df["predicted_fault"],
    average="macro"
)

print("\nFAULT TYPE")
print("-" * 70)
print(f"Correct            : {fault_correct.sum()}")
print(f"Incorrect          : {total - fault_correct.sum()}")
print(f"Accuracy           : {fault_accuracy:.2f}%")
print(f"Macro F1           : {fault_f1:.4f}")

# --------------------------------------------------
# COMBINED
# --------------------------------------------------

combined_correct = root_correct & fault_correct

combined_accuracy = combined_correct.mean() * 100

print("\nCOMBINED PERFORMANCE")
print("-" * 70)
print("A case is correct only when BOTH:")
print("  1. Root-cause service is correct")
print("  2. Fault type is correct")
print()
print(f"Correct            : {combined_correct.sum()}")
print(f"Incorrect          : {total - combined_correct.sum()}")
print(f"Combined Accuracy  : {combined_accuracy:.2f}%")

# --------------------------------------------------
# FAULT CLASSIFICATION REPORT
# --------------------------------------------------

print("\n" + "=" * 70)
print("FAULT TYPE CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        df["actual_fault"],
        df["predicted_fault"],
        digits=4
    )
)

# --------------------------------------------------
# FAULT CONFUSION COUNTS
# --------------------------------------------------

print("=" * 70)
print("FAULT TYPE CONFUSION TABLE")
print("=" * 70)

confusion = pd.crosstab(
    df["actual_fault"],
    df["predicted_fault"],
    rownames=["Actual"],
    colnames=["Predicted"]
)

print(confusion)

# --------------------------------------------------
# SAVE UPDATED RESULTS
# --------------------------------------------------

df["root_cause_correct"] = root_correct
df["fault_correct"] = fault_correct
df["combined_correct"] = combined_correct

df.to_csv(
    "data/api_evaluation_results_detailed.csv",
    index=False
)

print("\nDetailed results saved to:")
print("data/api_evaluation_results_detailed.csv")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)