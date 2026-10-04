from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# ============================================================
# 1. FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Root Cause Analysis API",
    description="AI-based Root Cause Analysis for Microservices",
    version="1.0.0"
)
FastAPIInstrumentor.instrument_app(app)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus monitoring
Instrumentator().instrument(app).expose(app)


# ============================================================
# 2. LOAD TRAINED MODELS
# ============================================================

RCA_MODEL_PATH = "models/xgboost_rca_model.pkl"
FAULT_MODEL_PATH = "models/xgboost_fault_model_v2.pkl"


try:

    rca_package = joblib.load(RCA_MODEL_PATH)
    fault_package = joblib.load(FAULT_MODEL_PATH)

    # RCA model
    rca_model = rca_package["model"]
    rca_features = rca_package["features"]

    # Fault model
    fault_model = fault_package["model"]
    fault_features = fault_package["feature_columns"]
    fault_encoder = fault_package["label_encoder"]

    print("RCA model loaded successfully!")
    print("Fault model loaded successfully!")

except Exception as e:

    print("Error loading models:", e)
    raise


# ============================================================
# 3. SERVICE FEATURE INPUT
# ============================================================

class ServiceFeatures(BaseModel):

    service: str

    cpu_change: float = 0.0
    cpu_ratio: float = 0.0

    mem_change: float = 0.0
    mem_ratio: float = 0.0

    diskio_change: float = 0.0
    diskio_ratio: float = 0.0

    socket_change: float = 0.0
    socket_ratio: float = 0.0

    workload_change: float = 0.0
    workload_ratio: float = 0.0

    error_change: float = 0.0
    error_ratio: float = 0.0

    # API uses "_" because "-" is not valid in Python variable names.
    # These are converted to the original training names later.

    latency_50_change: float = 0.0
    latency_50_ratio: float = 0.0

    latency_90_change: float = 0.0
    latency_90_ratio: float = 0.0

    log_count_change: float = 0.0
    log_count_ratio: float = 0.0

    log_error_count: float = 0.0
    log_timeout_count: float = 0.0
    log_connection_count: float = 0.0

    trace_count_change: float = 0.0
    trace_count_ratio: float = 0.0

    trace_duration_change: float = 0.0
    trace_duration_ratio: float = 0.0

    trace_failure_count: float = 0.0


# ============================================================
# 4. INCIDENT INPUT
# ============================================================

class PredictionRequest(BaseModel):

    application: str

    services: list[ServiceFeatures]


# ============================================================
# 5. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI Root Cause Analysis API is running",
        "status": "healthy"
    }


# ============================================================
# 6. MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {

        "root_cause_model": "XGBoost",

        "fault_type_model": "XGBoost",

        "root_cause_features": len(rca_features),

        "fault_type_features": len(fault_features),

        "fault_types": list(
            fault_encoder.classes_
        )
    }


# ============================================================
# 7. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if len(request.services) == 0:

            raise HTTPException(
                status_code=400,
                detail="At least one service is required."
            )


        # ----------------------------------------------------
        # Convert Pydantic objects to DataFrame
        # ----------------------------------------------------

        service_records = [
            service.model_dump()
            for service in request.services
        ]

        df = pd.DataFrame(
            service_records
        )


        # ----------------------------------------------------
        # Match API feature names with training feature names
        # ----------------------------------------------------

        df = df.rename(
            columns={
                "latency_50_change":
                    "latency-50_change",

                "latency_50_ratio":
                    "latency-50_ratio",

                "latency_90_change":
                    "latency-90_change",

                "latency_90_ratio":
                    "latency-90_ratio"
            }
        )


        # ----------------------------------------------------
        # Save service names
        # ----------------------------------------------------

        service_names = df["service"].tolist()


        # ====================================================
        # ROOT-CAUSE SERVICE PREDICTION
        # ====================================================

        # Check that all required RCA features exist

        missing_rca_features = [
            feature
            for feature in rca_features
            if feature not in df.columns
        ]

        if missing_rca_features:

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing RCA features",
                    "features": missing_rca_features
                }
            )


        # Select RCA features

        X_rca = df[
            rca_features
        ].copy()


        # Replace invalid values

        X_rca = X_rca.replace(
            [float("inf"), float("-inf")],
            0
        )

        X_rca = X_rca.fillna(0)


        # ----------------------------------------------------
        # Predict probability for each service
        # ----------------------------------------------------

        probabilities = (
            rca_model.predict_proba(X_rca)[:, 1]
        )


        # ----------------------------------------------------
        # Create service ranking
        # ----------------------------------------------------

        ranking = []

        for service, probability in zip(
            service_names,
            probabilities
        ):

            ranking.append({

                "service": service,

                "probability":
                    float(probability)
            })


        # Highest probability first

        ranking.sort(
            key=lambda x: x["probability"],
            reverse=True
        )


        # ----------------------------------------------------
        # Select root cause
        # ----------------------------------------------------

        root_cause = ranking[0]

        root_cause_service = (
            root_cause["service"]
        )

        root_cause_confidence = (
            root_cause["probability"]
        )


        # ----------------------------------------------------
        # Find root-cause service row
        # ----------------------------------------------------

        root_index = service_names.index(
            root_cause_service
        )

        root_row = df.iloc[
            [root_index]
        ].copy()


        # ====================================================
        # FAULT TYPE PREDICTION
        # ====================================================

        missing_fault_features = [
            feature
            for feature in fault_features
            if feature not in root_row.columns
        ]

        if missing_fault_features:

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing fault features",
                    "features": missing_fault_features
                }
            )


        X_fault = root_row[
            fault_features
        ].copy()


        X_fault = X_fault.replace(
            [float("inf"), float("-inf")],
            0
        )

        X_fault = X_fault.fillna(0)


        # ----------------------------------------------------
        # Predict fault type
        # ----------------------------------------------------

        fault_probabilities = (
            fault_model.predict_proba(
                X_fault
            )[0]
        )


        fault_prediction = (
            fault_model.predict(
                X_fault
            )[0]
        )


        fault_type = (
            fault_encoder.inverse_transform(
                [fault_prediction]
            )[0]
        )


        fault_confidence = float(
            max(fault_probabilities)
        )


        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        return {

            "application":
                request.application,

            "root_cause_service":
                root_cause_service,

            "fault_type":
                fault_type,

            "root_cause_confidence":
                round(
                    float(
                        root_cause_confidence
                    ),
                    4
                ),

            "fault_confidence":
                round(
                    fault_confidence,
                    4
                ),

            "service_ranking": [

                {
                    "service":
                        item["service"],

                    "probability":
                        round(
                            item["probability"],
                            4
                        )
                }

                for item in ranking
            ],

            "status":
                "prediction_successful"
        }


    except HTTPException:
        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )