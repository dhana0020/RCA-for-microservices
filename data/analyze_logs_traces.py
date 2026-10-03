import pandas as pd
from pathlib import Path

case_path = Path("data/re2_sample/re2ob_checkoutservice_cpu_1")

with open(case_path / "inject_time.txt") as f:
    inject_time = int(f.read().strip())

logs = pd.read_parquet(case_path / "logs.parquet")
traces = pd.read_parquet(case_path / "traces.parquet")


# ============================================================
# LOGS
# ============================================================

print("\n========== LOG TIMELINE ==========")

logs_before = logs[logs["timestamp"] < inject_time]
logs_after = logs[logs["timestamp"] >= inject_time]

print("Logs before:", len(logs_before))
print("Logs after :", len(logs_after))

print("\nLog counts by service:")
print(
    logs_after["container_name"]
    .value_counts()
    .head(15)
)

print("\nTop log messages after injection:")
print(
    logs_after["message"]
    .value_counts()
    .head(20)
)


# ============================================================
# LOG KEYWORDS
# ============================================================

print("\n========== LOG KEYWORDS ==========")

keywords = [
    "error",
    "exception",
    "failed",
    "failure",
    "timeout",
    "timed out",
    "connection",
    "unavailable"
]

for keyword in keywords:
    count = logs_after["message"].astype(str).str.contains(
        keyword,
        case=False,
        na=False
    ).sum()

    if count > 0:
        print(f"{keyword:15s}: {count}")


# ============================================================
# TRACES
# ============================================================

print("\n========== TRACE TIMELINE ==========")

# startTimeMillis is milliseconds.
# Convert milliseconds → seconds.
traces["timestamp_sec"] = (
    pd.to_numeric(
        traces["startTimeMillis"],
        errors="coerce"
    ) / 1000
)

traces = traces.dropna(subset=["timestamp_sec"])

traces_before = traces[
    traces["timestamp_sec"] < inject_time
]

traces_after = traces[
    traces["timestamp_sec"] >= inject_time
]

print("Traces before:", len(traces_before))
print("Traces after :", len(traces_after))


# ============================================================
# TRACE COUNTS
# ============================================================

print("\nTrace counts by service after injection:")

print(
    traces_after["serviceName"]
    .value_counts()
    .head(15)
)


# ============================================================
# TRACE DURATION
# ============================================================

print("\n========== TRACE DURATION ==========")

before_duration = (
    traces_before
    .groupby("serviceName")["duration"]
    .mean()
)

after_duration = (
    traces_after
    .groupby("serviceName")["duration"]
    .mean()
)

comparison = pd.DataFrame({
    "before_mean_duration": before_duration,
    "after_mean_duration": after_duration
})

comparison["change"] = (
    comparison["after_mean_duration"]
    - comparison["before_mean_duration"]
)

print(
    comparison
    .sort_values("change", ascending=False)
    .head(15)
)


# ============================================================
# TRACE STATUS
# ============================================================

print("\n========== TRACE STATUS CODES AFTER INJECTION ==========")

print(
    traces_after["statusCode"]
    .value_counts(dropna=False)
    .head(10)
)