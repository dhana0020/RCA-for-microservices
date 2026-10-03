import pandas as pd
from pathlib import Path

case_path = Path("data/re2_sample/re2ob_checkoutservice_cpu_1")

metrics = pd.read_parquet(case_path / "metrics.parquet")

with open(case_path / "inject_time.txt") as f:
    inject_time = int(f.read().strip())

print("Injection time:", inject_time)
print("Metric time range:", metrics["time"].min(), "to", metrics["time"].max())

# Split normal and faulty periods
normal = metrics[metrics["time"] < inject_time]
faulty = metrics[metrics["time"] >= inject_time]

print("\n========== TIMELINE ==========")
print("Normal rows:", len(normal))
print("Faulty rows:", len(faulty))

# Focus on checkoutservice
cols = [
    "checkoutservice_cpu",
    "checkoutservice_mem",
    "checkoutservice_workload",
    "checkoutservice_latency-50",
    "checkoutservice_latency-90"
]

print("\n========== CHECKOUTSERVICE ==========")

for col in cols:
    print(f"\n{col}")
    print("Normal mean :", normal[col].mean())
    print("Faulty mean :", faulty[col].mean())
    print("Normal max  :", normal[col].max())
    print("Faulty max  :", faulty[col].max())

print("\n========== ALL SERVICES CPU ==========")

cpu_cols = [c for c in metrics.columns if c.endswith("_cpu")]

for col in cpu_cols:
    print(
        f"{col:35s} "
        f"normal={normal[col].mean():.4f} "
        f"faulty={faulty[col].mean():.4f}"
    )