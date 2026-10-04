import mlflow
import mlflow.xgboost
import joblib
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

RCA_MODEL_PATH = "models/xgboost_rca_model.pkl"
FAULT_MODEL_PATH = "models/xgboost_fault_model_v2.pkl"

EXPERIMENT_NAME = "AI-RCA-Microservices"


# ============================================================
# CREATE / SELECT EXPERIMENT
# ============================================================

mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# LOAD MODELS
# ============================================================

rca_package = joblib.load(RCA_MODEL_PATH)
fault_package = joblib.load(FAULT_MODEL_PATH)

rca_model = rca_package["model"]
fault_model = fault_package["model"]


# ============================================================
# LOAD EVALUATION RESULTS
# ============================================================

results_path = "data/api_evaluation_results_detailed.csv"

df = pd.read_csv(results_path)


# ============================================================
# CALCULATE FINAL METRICS
# ============================================================

root_accuracy = (
    df["actual_root_cause"] ==
    df["predicted_root_cause"]
).mean()

fault_accuracy = (
    df["actual_fault"] ==
    df["predicted_fault"]
).mean()

combined_accuracy = (
    df["combined_correct"]
).mean()


# ============================================================
# MLflow RUN
# ============================================================

with mlflow.start_run(run_name="RCA_XGBoost_Final"):

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    mlflow.log_param(
        "rca_model",
        "XGBoost"
    )

    mlflow.log_param(
        "fault_model",
        "XGBoost"
    )

    mlflow.log_param(
        "dataset",
        "RCAEval RE2"
    )

    mlflow.log_param(
        "total_cases",
        len(df)
    )

    mlflow.log_param(
        "rca_features",
        len(rca_package["features"])
    )

    mlflow.log_param(
        "fault_features",
        len(fault_package["feature_columns"])
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mlflow.log_metric(
        "root_cause_accuracy",
        root_accuracy
    )

    mlflow.log_metric(
        "fault_accuracy",
        fault_accuracy
    )

    mlflow.log_metric(
        "combined_accuracy",
        combined_accuracy
    )

    # --------------------------------------------------------
    # Log models
    # --------------------------------------------------------

    mlflow.xgboost.log_model(
        rca_model,
        name="rca_model"
    )

    mlflow.xgboost.log_model(
        fault_model,
        name="fault_model"
    )

    # --------------------------------------------------------
    # Log evaluation file
    # --------------------------------------------------------

    mlflow.log_artifact(
        results_path,
        artifact_path="evaluation"
    )

    print("=" * 70)
    print("MLFLOW RUN COMPLETED")
    print("=" * 70)

    print(f"Root Cause Accuracy : {root_accuracy * 100:.2f}%")
    print(f"Fault Accuracy      : {fault_accuracy * 100:.2f}%")
    print(f"Combined Accuracy   : {combined_accuracy * 100:.2f}%")

    print("\nExperiment:")
    print(EXPERIMENT_NAME)

    print("\nModels logged:")
    print("- RCA XGBoost")
    print("- Fault XGBoost V2")