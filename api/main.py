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
    allow_origins=[
        "https://ai-rca-dashboard-dhana.onrender.com",
        "http://localhost:5173",
    ],
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
