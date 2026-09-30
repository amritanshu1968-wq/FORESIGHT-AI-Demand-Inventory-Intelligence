# PROJECT FORESIGHT — PHASE 7: FASTAPI SCORING SERVICE REPORT

> **DISCLAIMER & PROVENANCE NOTICE**  
> **Data Source:** Synthetic retail dataset generated for project development and demonstration. These findings demonstrate the capabilities of the FORESIGHT platform and should not be interpreted as actual NorthBay Living or Zidio Development commercial results.

---

## 1. Executive Summary

Phase 7 delivers an enterprise-grade, low-latency REST inference service for Project FORESIGHT using **FastAPI** and **Uvicorn**. The API serves as a stateless operational layer that bridges the validated forecasting artifacts (`models/production_forecaster.joblib`, `outputs/forecasts/forecast_results.csv`) and the 2D inventory risk engine (`outputs/risk/risk_register.csv`) to external ERP systems, automated procurement bots, and warehouse management systems (WMS).

### Core Architectural Principles
1. **Zero Pipeline Mutation:** The service operates strictly in inference/scoring mode. It does not recompute pipelines, retrain models, or generate random synthetic data.
2. **Artifact-Driven & Cached:** Artifacts and models are loaded once into memory during application lifespan startup to reduce repeated file/model loading during requests.
3. **Strict Type Safety:** All payloads, query parameters, and responses are strongly validated using Pydantic v2 schemas (`api/schemas.py`).
4. **Transparent Data Provenance:** Every response payload embeds the mandatory synthetic data disclosure tag.

---

## 2. API Endpoints Specification

| Method | Endpoint | Description | Status Codes |
|---|---|---|---|
| `GET` | `/` | Service root with API metadata, documentation links, and synthetic disclaimer | 200 |
| `GET` | `/health` | Health check verifying model status, artifact availability, and loaded SKU count | 200, 503 |
| `GET` | `/forecast/{sku_id}` | 12-week forward demand forecast with 80% empirical prediction intervals | 200, 404 |
| `GET` | `/risk-score/{sku_id}` | Real-time 2D inventory risk scoring, quadrant assignment, and ₹ exposure | 200, 404 |
| `GET` | `/sku/{sku_id}` | Comprehensive 360-degree SKU profile (master, inventory, forecast, risk, impact) | 200, 404 |
| `POST` | `/batch-score` | Multi-SKU bulk scoring endpoint for procurement batch execution | 200, 422 |
| `GET` | `/docs` | Interactive Swagger UI documentation | 200 |
| `GET` | `/redoc` | Interactive ReDoc technical documentation | 200 |

---

## 3. Production Schemas & Response Structures

### 3.1 Health Check (`GET /health`)
```json
{
  "status": "healthy",
  "service": "FORESIGHT Scoring API",
  "model_loaded": true,
  "forecast_artifact_loaded": true,
  "risk_artifact_loaded": true,
  "total_skus": 300,
  "data_disclaimer": "Data Source: Synthetic retail dataset generated for project development and demonstration. These findings demonstrate the capabilities of the FORESIGHT platform and should not be interpreted as actual NorthBay Living or Zidio Development business results."
}
```

### 3.2 SKU Forecast (`GET /forecast/SKU-0272`)
```json
{
  "sku_id": "SKU-0272",
  "category": "Small Appliances",
  "model_name": "hist_gbm",
  "forecast_horizon_weeks": 12,
  "forecast": [
    {
      "week": "2026-01-05",
      "horizon_step": 1,
      "forecast_units": 76.77,
      "lower_bound": 58.18,
      "upper_bound": 95.36
    },
    ...
  ],
  "data_disclaimer": "Data Source: Synthetic retail dataset generated for project development and demonstration. ..."
}
```

### 3.3 SKU 2D Risk Score (`GET /risk-score/SKU-0272`)
```json
{
  "sku_id": "SKU-0272",
  "product_name": "NorthBay Studio Cold Press Juicers #272",
  "category": "Small Appliances",
  "decision_quadrant": "Reorder Now",
  "stockout_risk_score": 0.6413,
  "overstock_risk_score": 0.0,
  "current_on_hand": 0,
  "on_order_units": 113,
  "weeks_of_cover": 0.0,
  "safety_stock": 97,
  "reorder_point": 255,
  "recommended_order_qty": 242,
  "excess_units": 0,
  "lost_margin_risk": 196840.41,
  "excess_capital_locked": 0.0,
  "action_priority": "HIGH",
  "recommended_action": "Urgent replenishment order of 242 units needed to avoid stockout.",
  "data_disclaimer": "Data Source: Synthetic retail dataset generated for project development and demonstration. ..."
}
```

### 3.4 Multi-SKU Batch Scoring (`POST /batch-score`)
**Request Body:**
```json
{
  "sku_ids": ["SKU-0272", "SKU-0280", "SKU-0036"]
}
```
**Response (Truncated for brevity):**
```json
{
  "count": 3,
  "results": [
    {
      "sku_id": "SKU-0272",
      "decision_quadrant": "Reorder Now",
      "action_priority": "HIGH",
      "recommended_order_qty": 242,
      "lost_margin_risk": 196840.41,
      "excess_capital_locked": 0.0
    },
    {
      "sku_id": "SKU-0280",
      "decision_quadrant": "Reorder Now",
      "action_priority": "HIGH",
      "recommended_order_qty": 281,
      "lost_margin_risk": 172782.71,
      "excess_capital_locked": 0.0
    },
    {
      "sku_id": "SKU-0036",
      "decision_quadrant": "Watch / Volatile",
      "action_priority": "HIGH",
      "recommended_order_qty": 0,
      "lost_margin_risk": 0.0,
      "excess_capital_locked": 544743.78
    }
  ],
  "unmatched_sku_ids": [],
  "data_disclaimer": "Data Source: Synthetic retail dataset generated for project development and demonstration. ..."
}
```

---

## 4. Verification & Testing

### 4.1 Pytest Unit Test Suite (`tests/test_api.py`)
10 dedicated API test cases covering:
1. `test_root_endpoint`: Verifies root metadata, service version, and synthetic disclaimer.
2. `test_health_endpoint`: Asserts HTTP 200 and all artifacts loaded (`model_loaded=True`, `total_skus=300`).
3. `test_forecast_valid_sku`: Asserts 12-week forecast array, bounds sanity ($lower \le forecast \le upper$).
4. `test_forecast_invalid_sku`: Asserts proper HTTP 404 and error detail for `SKU-9999`.
5. `test_risk_score_valid_sku`: Asserts numeric risk scores $[0.0, 1.0]$, valid decision quadrant, and rupee exposures $\ge 0$.
6. `test_risk_score_invalid_sku`: Asserts proper HTTP 404 for unindexed SKU.
7. `test_sku_combined_view`: Asserts nested product, inventory, forecast, risk, and action payloads.
8. `test_sku_combined_invalid`: Asserts HTTP 404.
9. `test_batch_score_endpoint`: Validates batch array scoring and handling of empty or mixed lists.
10. `test_batch_score_empty_request`: Asserts HTTP 422 Unprocessable Entity on validation violations.

**Result:** `10 passed in 0.45s` (Overall test suite: `32 passed in 3.79s`).

### 4.2 Live Smoke Test Execution (`scripts/smoke_test_api.py`)
A live Uvicorn daemon was launched on `127.0.0.1:8000`. Real HTTP requests were transmitted using standard client libraries:
- `GET /` $\rightarrow$ `HTTP 200 OK`
- `GET /health` $\rightarrow$ `HTTP 200 OK`
- `GET /forecast/SKU-0272` $\rightarrow$ `HTTP 200 OK` (12 steps returned)
- `GET /risk-score/SKU-0272` $\rightarrow$ `HTTP 200 OK` (`Reorder Now`, 242 units recommended)
- `GET /sku/SKU-0272` $\rightarrow$ `HTTP 200 OK` (Complete 360-degree view)
- `POST /batch-score` $\rightarrow$ `HTTP 200 OK` (3 items scored)
- `GET /forecast/SKU-9999` $\rightarrow$ `HTTP 404 Not Found` (Graceful exception handling)

---

## 5. Deployment & Execution Instructions

### Local Development / Evaluation:
```bash
# 1. Start the API service
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# 2. In another terminal, run the smoke test
python scripts/smoke_test_api.py

# 3. Access Swagger UI docs
# Open http://127.0.0.1:8000/docs
```

### Production Deployment (Gunicorn / Uvicorn Workers):
```bash
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
