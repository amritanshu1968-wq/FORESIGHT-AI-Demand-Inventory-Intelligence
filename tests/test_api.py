import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "FORESIGHT" in data["service"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["forecast_artifact_loaded"] is True
    assert data["risk_artifact_loaded"] is True
    assert data["total_skus"] == 300


def test_forecast_existing_sku():
    response = client.get("/forecast/SKU-0272")
    assert response.status_code == 200
    data = response.json()
    assert data["sku_id"] == "SKU-0272"
    assert data["forecast_horizon_weeks"] == 12
    assert len(data["forecast"]) == 12
    first_pt = data["forecast"][0]
    assert "week" in first_pt
    assert "forecast_units" in first_pt
    assert "lower_bound" in first_pt
    assert "upper_bound" in first_pt
    assert first_pt["lower_bound"] <= first_pt["forecast_units"] <= first_pt["upper_bound"]


def test_forecast_unknown_sku_returns_404():
    response = client.get("/forecast/SKU-9999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_risk_score_existing_sku():
    response = client.get("/risk-score/SKU-0272")
    assert response.status_code == 200
    data = response.json()
    assert data["sku_id"] == "SKU-0272"
    assert 0.0 <= data["stockout_risk_score"] <= 1.0
    assert 0.0 <= data["overstock_risk_score"] <= 1.0
    assert data["decision_quadrant"] in ["Reorder Now", "Markdown / Clear", "Watch / Volatile", "Healthy"]
    assert data["lost_margin_risk"] >= 0.0
    assert data["excess_capital_locked"] >= 0.0
    assert data["recommended_order_qty"] >= 0


def test_risk_score_unknown_sku_returns_404():
    response = client.get("/risk-score/SKU-9999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_sku_decision_card():
    response = client.get("/sku/SKU-0272")
    assert response.status_code == 200
    data = response.json()
    assert data["sku_id"] == "SKU-0272"
    assert "product" in data
    assert "inventory" in data
    assert "forecast_summary" in data
    assert "risk" in data
    assert "financial_impact" in data
    assert "action" in data
    assert data["product"]["unit_cost"] < data["product"]["unit_price"]


def test_batch_score_endpoint():
    payload = {"sku_ids": ["SKU-0272", "SKU-0280", "SKU-0036"]}
    response = client.post("/batch-score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["results"]) == 3
    assert len(data["unmatched_sku_ids"]) == 0


def test_batch_score_with_unmatched():
    payload = {"sku_ids": ["SKU-0272", "SKU-UNKNOWN-999"]}
    response = client.post("/batch-score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert "SKU-UNKNOWN-999" in data["unmatched_sku_ids"]


def test_batch_score_empty_rejected():
    payload = {"sku_ids": []}
    response = client.post("/batch-score", json=payload)
    assert response.status_code == 422  # Pydantic validation error for empty list
