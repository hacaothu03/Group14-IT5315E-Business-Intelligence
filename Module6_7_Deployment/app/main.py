"""FastAPI service for the house-price valuation model (Module 6).

Endpoints
---------
- ``GET  /``            : service metadata + links
- ``GET  /health``      : liveness / readiness probe
- ``GET  /model-info``  : model version, transforms, required fields
- ``POST /predict``     : single valuation with a confidence interval
- ``POST /predict/batch``: up to 1000 valuations in one call
- ``GET  /docs``        : interactive OpenAPI docs (Swagger UI)

Run locally::

    uvicorn app.main:app --reload --port 8000   # from Module6_7_Deployment/

The model is loaded once at startup and reused across requests. Every
prediction is logged to a JSONL file (``PREDICTION_LOG_PATH``) for Module 7.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.model_service import get_service
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    PropertyFeatures,
)

service = get_service()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the model eagerly so the first request is fast and startup fails
    # loudly if artifacts are missing.
    service.warm_up()
    yield


app = FastAPI(
    title="House Price Valuation API",
    version="1.0.0",
    description=(
        "Estimates the market sale price of a property and returns a confidence "
        "interval. Wraps the Module 4-5 Lasso pipeline (model version `lasso_v1`). "
        "Not a legal appraisal."
    ),
    lifespan=lifespan,
)

# Allow the Streamlit demo (and other browser clients) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    loaded = service._artifacts is not None
    return HealthResponse(
        status="ok" if loaded else "starting",
        model_loaded=loaded,
        model_version=service.artifacts["model_version"] if loaded else None,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**service.model_info())


@app.post("/predict", response_model=PredictionResponse, tags=["valuation"])
def predict(
    features: PropertyFeatures,
    confidence_level: Optional[float] = Query(
        None, ge=0.50, le=0.999, description="Interval coverage (default 0.80)."
    ),
) -> PredictionResponse:
    result = service.predict(features.to_raw_dict(), confidence_level=confidence_level)
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["valuation"])
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    rows = [item.to_raw_dict() for item in request.items]
    results = service.predict_batch(rows, confidence_level=request.confidence_level)
    return BatchPredictResponse(
        count=len(results),
        predictions=[PredictionResponse(**r) for r in results],
    )
