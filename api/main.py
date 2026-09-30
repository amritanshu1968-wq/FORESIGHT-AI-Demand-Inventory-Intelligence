

from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    ForecastResponse,
    HealthResponse,
    RiskScoreResponse,
    SKUDecisionResponse,
)
from api.services.forecast_service import ForecastService
from api.services.risk_service import RiskScoringService
from config import DATASET_DISCLAIMER
from src.utils.logger import get_logger

logger = get_logger("api_main")

# Initialize FastAPI application with rich OpenAPI metadata
app = FastAPI(
    title="Project FORESIGHT — Demand & Inventory Intelligence API",
    description=(
        "Production-grade RESTful scoring service for NorthBay Living D2C operations. "
        "Provides SKU-level forward demand forecasting, 2D inventory risk classification, "
        "reorder point guidance, and working capital optimization.\n\n"
        f"**Governance Disclosure**: {DATASET_DISCLAIMER}"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware (Restricted to standard local dashboard and frontend ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global Service Instances (Loaded once on startup)
forecast_service = ForecastService()
risk_service = RiskScoringService()


@app.get(
    "/",
    tags=["General"],
    summary="API Root & Information",
)
def root():
    """Returns basic service metadata and documentation links."""
    return {
        "service": "Project FORESIGHT — Demand & Inventory Intelligence API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health",
        "data_disclaimer": DATASET_DISCLAIMER,
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="System Health & Artifact Audit",
)
def health_check():
    """
    Dynamically reports the health of the API and verifies whether required
    production models and precomputed artifacts are successfully loaded.
    """
    is_model = forecast_service.is_model_loaded()
    is_fc = forecast_service.is_loaded()
    is_risk = risk_service.is_loaded()

    all_healthy = is_model and is_fc and is_risk
    status_str = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=status_str,
        service="FORESIGHT Scoring API",
        model_loaded=is_model,
        forecast_artifact_loaded=is_fc,
        risk_artifact_loaded=is_risk,
        total_skus=risk_service.get_total_skus(),
        data_disclaimer=DATASET_DISCLAIMER,
    )


@app.get(
    "/forecast/{sku_id}",
    response_model=ForecastResponse,
    tags=["Forecasting"],
    summary="Retrieve 12-Week Demand Forecast for SKU",
)
def get_sku_forecast(sku_id: str):
    """
    Returns the validated 12-week forward demand forecast for a specified SKU,
    including point estimates and calibrated 80% empirical prediction intervals.
    """
    if not forecast_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecasting artifact unavailable. Please run the pipeline first.",
        )

    clean_sku = sku_id.strip()
    result = forecast_service.get_forecast(clean_sku)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"SKU {clean_sku} not found",
        )

    return ForecastResponse(**result)


@app.get(
    "/risk-score/{sku_id}",
    response_model=RiskScoreResponse,
    tags=["Inventory Risk"],
    summary="Retrieve 2D Inventory Risk Scores for SKU",
)
def get_sku_risk_score(sku_id: str):
    """
    Returns normalized stockout and overstock risk scores, weeks of cover,
    ROP, safety stock, recommended batch quantities, and rupee exposure.
    """
    if not risk_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk register artifact unavailable. Please run the pipeline first.",
        )

    clean_sku = sku_id.strip()
    result = risk_service.get_risk_score(clean_sku)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"SKU {clean_sku} not found",
        )

    return RiskScoreResponse(**result)


@app.get(
    "/sku/{sku_id}",
    response_model=SKUDecisionResponse,
    tags=["Unified Decisioning"],
    summary="Retrieve Unified SKU Profile & Decision Card",
)
def get_sku_decision_card(sku_id: str):
    """
    Combines product economics, warehouse inventory levels, demand forecast summary,
    2D risk scores, and recommended operational actions into a unified response.
    """
    if not risk_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk register artifact unavailable. Please run the pipeline first.",
        )

    clean_sku = sku_id.strip()
    result = risk_service.get_sku_decision(clean_sku)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"SKU {clean_sku} not found",
        )

    return SKUDecisionResponse(**result)


@app.post(
    "/batch-score",
    response_model=BatchScoreResponse,
    tags=["Inventory Risk"],
    summary="Batch Score Multiple SKUs",
)
def batch_score_skus(payload: BatchScoreRequest):
    """
    Accepts a list of SKU IDs and returns operational risk classifications,
    recommended batch order sizes, and financial exposure in a single call.
    """
    if not risk_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk register artifact unavailable. Please run the pipeline first.",
        )

    if not payload.sku_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list of sku_ids cannot be empty.",
        )

    results, unmatched = risk_service.batch_score(payload.sku_ids)

    return BatchScoreResponse(
        count=len(results),
        results=results,
        unmatched_sku_ids=unmatched,
        data_disclaimer=DATASET_DISCLAIMER,
    )
