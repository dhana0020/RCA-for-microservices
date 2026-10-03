
import pandas as pd
from pathlib import Path


# ================================================================
# SAFE RELATIVE CHANGE
# ================================================================

def safe_change(before, after):

    if abs(before) < 1e-8:
        return 0.0

    return (after - before) / abs(before)


# ================================================================
# EXTRACT FEATURES FOR ONE CASE
# ================================================================

def extract_case_features(case_path):

    case_path = Path(case_path)

    # ============================================================
    # LOAD METRICS
    # Metrics are required
    # ============================================================

    metrics = pd.read_parquet(
        case_path / "metrics.parquet"
    )

    # ============================================================
    # LOAD LOGS
    # Logs may be missing
    # ============================================================

    logs_file = case_path / "logs.parquet"

    if logs_file.exists():

        logs = pd.read_parquet(
            logs_file
        )

    else:

        logs = pd.DataFrame(
            columns=[
                "timestamp",
                "container_name",
                "message"
            ]
        )

    # ============================================================
    # LOAD TRACES
    # Traces may be missing
    # ============================================================

    traces_file = case_path / "traces.parquet"

    if traces_file.exists():

        traces = pd.read_parquet(
            traces_file
        )

    else:

        traces = pd.DataFrame(
            columns=[
                "startTimeMillis",
                "serviceName",
                "duration",
                "statusCode"
            ]
        )

    # ============================================================
    # INJECTION TIME
    # ============================================================

    with open(
        case_path / "inject_time.txt",
        "r"
    ) as f:

        inject_time = int(
            f.read().strip()
        )

    # ============================================================
    # SPLIT METRICS
    # ============================================================

    metrics_before = metrics[
        metrics["time"] < inject_time
    ]

    metrics_after = metrics[
        metrics["time"] >= inject_time
    ]

    # ============================================================
    # SPLIT LOGS
    # ============================================================

    if len(logs) > 0:

        logs_before = logs[
            logs["timestamp"] < inject_time
        ]

        logs_after = logs[
            logs["timestamp"] >= inject_time
        ]

    else:

        logs_before = logs.copy()
        logs_after = logs.copy()

    # ============================================================
    # SPLIT TRACES
    # ============================================================

    if len(traces) > 0:

        # startTimeMillis = milliseconds
        # inject_time = seconds

        traces["timestamp_sec"] = (
            pd.to_numeric(
                traces["startTimeMillis"],
                errors="coerce"
            ) / 1000
        )

        traces = traces.dropna(
            subset=["timestamp_sec"]
        )

        traces_before = traces[
            traces["timestamp_sec"] < inject_time
        ]

        traces_after = traces[
            traces["timestamp_sec"] >= inject_time
        ]

    else:

        traces_before = traces.copy()
        traces_after = traces.copy()

    # ============================================================
    # FIND SERVICES FROM METRICS
    # ============================================================

    services = set()

    for column in metrics.columns:

        for suffix in [
            "_cpu",
            "_mem",
            "_diskio",
            "_socket",
            "_workload",
            "_error",
            "_latency-50",
            "_latency-90"
        ]:

            if column.endswith(suffix):

                services.add(
                    column[:-len(suffix)]
                )

    # frontend-external is not a root-cause candidate

    services.discard(
        "frontend-external"
    )

    # ============================================================
    # EXTRACT FEATURES FOR EACH SERVICE
    # ============================================================

    rows = []

    for service in sorted(services):

        row = {
            "service": service
        }

        # ========================================================
        # METRIC FEATURES
        # ========================================================

        metric_types = [
            "cpu",
            "mem",
            "diskio",
            "socket",
            "workload",
            "error",
            "latency-50",
            "latency-90"
        ]

        for metric_type in metric_types:

            column = (
                f"{service}_{metric_type}"
            )

            # ----------------------------------------------------
            # Metric does not exist
            # ----------------------------------------------------

            if column not in metrics.columns:

                row[
                    f"{metric_type}_change"
                ] = 0.0

                row[
                    f"{metric_type}_ratio"
                ] = 0.0

                continue

            # ----------------------------------------------------
            # Before failure
            # ----------------------------------------------------

            before_mean = pd.to_numeric(
                metrics_before[column],
                errors="coerce"
            ).mean()

            # ----------------------------------------------------
            # After failure
            # ----------------------------------------------------

            after_mean = pd.to_numeric(
                metrics_after[column],
                errors="coerce"
            ).mean()

            if pd.isna(before_mean):
                before_mean = 0.0

            if pd.isna(after_mean):
                after_mean = 0.0

            before_mean = float(
                before_mean
            )

            after_mean = float(
                after_mean
            )

            # ----------------------------------------------------
            # Absolute change
            # ----------------------------------------------------

            row[
                f"{metric_type}_change"
            ] = (
                after_mean - before_mean
            )

            # ----------------------------------------------------
            # Relative change
            # ----------------------------------------------------

            row[
                f"{metric_type}_ratio"
            ] = safe_change(
                before_mean,
                after_mean
            )

        # ========================================================
        # LOG FEATURES
        # ========================================================

        if len(logs) > 0:

            service_logs_before = logs_before[
                logs_before["container_name"]
                == service
            ]

            service_logs_after = logs_after[
                logs_after["container_name"]
                == service
            ]

        else:

            service_logs_before = logs_before
            service_logs_after = logs_after

        # --------------------------------------------------------
        # Log count
        # --------------------------------------------------------

        before_log_count = len(
            service_logs_before
        )

        after_log_count = len(
            service_logs_after
        )

        row["log_count_change"] = (
            after_log_count
            - before_log_count
        )

        row["log_count_ratio"] = safe_change(
            before_log_count,
            after_log_count
        )

        # --------------------------------------------------------
        # Log messages
        # --------------------------------------------------------

        if len(service_logs_after) > 0:

            messages = (
                service_logs_after["message"]
                .astype(str)
                .str.lower()
            )

        else:

            messages = pd.Series(
                dtype="object"
            )

        # --------------------------------------------------------
        # Error messages
        # --------------------------------------------------------

        row["log_error_count"] = (
            messages.str.contains(
                "error|exception|failed|failure",
                regex=True,
                na=False
            ).sum()
        )

        # --------------------------------------------------------
        # Timeout messages
        # --------------------------------------------------------

        row["log_timeout_count"] = (
            messages.str.contains(
                "timeout|timed out",
                regex=True,
                na=False
            ).sum()
        )

        # --------------------------------------------------------
        # Connection messages
        # --------------------------------------------------------

        row["log_connection_count"] = (
            messages.str.contains(
                "connection|unavailable",
                regex=True,
                na=False
            ).sum()
        )

        # ========================================================
        # TRACE FEATURES
        # ========================================================

        if len(traces) > 0:

            service_traces_before = (
                traces_before[
                    traces_before["serviceName"]
                    == service
                ]
            )

            service_traces_after = (
                traces_after[
                    traces_after["serviceName"]
                    == service
                ]
            )

        else:

            service_traces_before = traces_before
            service_traces_after = traces_after

        # --------------------------------------------------------
        # Trace count
        # --------------------------------------------------------

        before_trace_count = len(
            service_traces_before
        )

        after_trace_count = len(
            service_traces_after
        )

        row["trace_count_change"] = (
            after_trace_count
            - before_trace_count
        )

        row["trace_count_ratio"] = safe_change(
            before_trace_count,
            after_trace_count
        )

        # --------------------------------------------------------
        # Trace duration
        # --------------------------------------------------------

        before_duration = pd.to_numeric(
            service_traces_before["duration"],
            errors="coerce"
        ).mean()

        after_duration = pd.to_numeric(
            service_traces_after["duration"],
            errors="coerce"
        ).mean()

        if pd.isna(before_duration):
            before_duration = 0.0

        if pd.isna(after_duration):
            after_duration = 0.0

        before_duration = float(
            before_duration
        )

        after_duration = float(
            after_duration
        )

        row["trace_duration_change"] = (
            after_duration
            - before_duration
        )

        row["trace_duration_ratio"] = safe_change(
            before_duration,
            after_duration
        )

        # ========================================================
        # TRACE FAILURE COUNT
        # ========================================================

        if len(service_traces_after) > 0:

            status_after = (
                service_traces_after["statusCode"]
                .astype(str)
            )

            row["trace_failure_count"] = (
                ~status_after.isin(
                    [
                        "0",
                        "nan",
                        "<NA>"
                    ]
                )
            ).sum()

        else:

            row["trace_failure_count"] = 0

        # ========================================================
        # ADD SERVICE ROW
        # ========================================================

        rows.append(row)

    # ============================================================
    # RETURN FEATURE TABLE
    # ============================================================

    return pd.DataFrame(rows)


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":

    case_path = (
        "data/re2_cases/"
        "re2ob_checkoutservice_cpu_1"
    )

    features = extract_case_features(
        case_path
    )

    print(
        "\n========== FEATURE TABLE =========="
    )

    print(
        "Shape:",
        features.shape
    )

    print(
        "\nColumns:"
    )

    print(
        features.columns.tolist()
    )

    print(
        "\nFeatures:"
    )

    print(
        features.to_string(
            index=False
        )
    )

