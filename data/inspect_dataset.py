import pandas as pd
from pathlib import Path

case_path = Path("data/re2_sample/re2ob_checkoutservice_cpu_1")

# Load files
metrics = pd.read_parquet(case_path / "metrics.parquet")
logs = pd.read_parquet(case_path / "logs.parquet")
traces = pd.read_parquet(case_path / "traces.parquet")

print("\n========== METRICS ==========")
print("Shape:", metrics.shape)
print("Columns:")
print(metrics.columns.tolist())
print("\nFirst 5 rows:")
print(metrics.head())

print("\n========== LOGS ==========")
print("Shape:", logs.shape)
print("Columns:")
print(logs.columns.tolist())
print("\nFirst 5 rows:")
print(logs.head())

print("\n========== TRACES ==========")
print("Shape:", traces.shape)
print("Columns:")
print(traces.columns.tolist())
print("\nFirst 5 rows:")
print(traces.head())

print("\n========== INJECTION TIME ==========")
with open(case_path / "inject_time.txt") as f:
    print(f.read())