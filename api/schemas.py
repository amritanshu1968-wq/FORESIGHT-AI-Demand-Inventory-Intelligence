from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status ('healthy' or 'degraded')")
    service: str = Field("FORESIGHT Scoring API", description="API Service Name")
    model_loaded: bool = Field(..., description="Whether production forecaster joblib is loaded")
    forecast_artifact_loaded: bool = Field(..., description="Whether forward forecast CSV is loaded")
    risk_artifact_loaded: bool = Field(..., description="Whether risk register CSV is loaded")
    total_skus: int = Field(..., description="Number of active scored SKUs")
    data_disclaimer: str = Field(..., description="Synthetic data governance disclosure")


class ForecastPoint(BaseModel):
    week: str = Field(..., description="Forecast calendar week start date (YYYY-MM-DD)")
    horizon_step: int = Field(..., description="Forward week index (1 to 12)")
    forecast_units: float = Field(..., description="Point demand forecast in units")
    lower_bound: float = Field(..., description="80% empirical prediction interval lower bound")
    upper_bound: float = Field(..., description="80% empirical prediction interval upper bound")


class ForecastResponse(BaseModel):
    sku_id: str = Field(..., description="Unique product SKU identifier")
    category: str = Field(..., description="Product category")
    model_name: str = Field(..., description="Forecasting model name")
    forecast_horizon_weeks: int = Field(..., description="Number of forward weeks forecasted")
    forecast: List[ForecastPoint] = Field(..., description="Weekly forecast series")
    data_disclaimer: str = Field(..., description="Synthetic data governance disclosure")


class RiskScoreResponse(BaseModel):
    sku_id: str = Field(..., description="Product SKU identifier")
    product_name: str = Field(..., description="Product catalog name")
    category: str = Field(..., description="Product category")
    decision_quadrant: str = Field(..., description="2D decision grid quadrant ('Reorder Now', 'Markdown / Clear', 'Watch / Volatile', 'Healthy')")
    stockout_risk_score: float = Field(..., ge=0.0, le=1.0, description="Normalized stockout risk score [0, 1]")
    overstock_risk_score: float = Field(..., ge=0.0, le=1.0, description="Normalized overstock risk score [0, 1]")
    current_on_hand: int = Field(..., ge=0, description="Current on-hand warehouse stock")
    on_order_units: int = Field(..., ge=0, description="In-flight purchase order units")
    weeks_of_cover: float = Field(..., ge=0.0, description="Projected weeks of cover")
    safety_stock: int = Field(..., ge=0, description="Calibrated safety stock threshold")
    reorder_point: int = Field(..., ge=0, description="Dynamic reorder point")
    recommended_order_qty: int = Field(..., ge=0, description="Recommended replenishment batch size")
    excess_units: int = Field(..., ge=0, description="Excess units eligible for markdown")
    lost_margin_risk: float = Field(..., ge=0.0, description="Gross margin at risk in INR")
    excess_capital_locked: float = Field(..., ge=0.0, description="Working capital locked in INR")
    action_priority: str = Field(..., description="Operational priority ('HIGH', 'MEDIUM', 'LOW')")
    recommended_action: str = Field(..., description="Operational action directive")
    data_disclaimer: str = Field(..., description="Synthetic data governance disclosure")


class ProductInfo(BaseModel):
    product_name: str
    category: str
    unit_price: float
    unit_cost: float
    gross_margin_pct: float
    lead_time_days: int


class InventoryInfo(BaseModel):
    current_on_hand: int
    on_order_units: int
    weeks_of_cover: float
    safety_stock: int
    reorder_point: int
    inventory_value: float


class RiskInfo(BaseModel):
    stockout_risk_score: float = Field(..., ge=0.0, le=1.0)
    overstock_risk_score: float = Field(..., ge=0.0, le=1.0)
    decision_quadrant: str


class FinancialImpact(BaseModel):
    lost_margin_risk: float = Field(..., ge=0.0)
    excess_capital_locked: float = Field(..., ge=0.0)
    total_financial_exposure: float = Field(..., ge=0.0)


class ActionInfo(BaseModel):
    action_priority: str
    recommended_order_qty: int = Field(..., ge=0)
    excess_units: int = Field(..., ge=0)
    recommended_action: str


class SKUDecisionResponse(BaseModel):
    sku_id: str
    product: ProductInfo
    inventory: InventoryInfo
    forecast_summary: Dict[str, float]
    risk: RiskInfo
    financial_impact: FinancialImpact
    action: ActionInfo
    data_disclaimer: str


class BatchScoreRequest(BaseModel):
    sku_ids: List[str] = Field(..., min_length=1, max_length=500, description="List of SKU IDs to score (1 to 500)")

    @field_validator("sku_ids")
    @classmethod
    def validate_non_empty_skus(cls, v: List[str]) -> List[str]:
        cleaned = [s.strip() for s in v if s and isinstance(s, str) and s.strip()]
        if not cleaned:
            raise ValueError("sku_ids must contain at least one valid non-empty string.")
        return cleaned


class BatchScoreItem(BaseModel):
    sku_id: str
    product_name: str
    category: str
    decision_quadrant: str
    action_priority: str
    stockout_risk_score: float = Field(..., ge=0.0, le=1.0)
    overstock_risk_score: float = Field(..., ge=0.0, le=1.0)
    recommended_order_qty: int = Field(..., ge=0)
    excess_units: int = Field(..., ge=0)
    lost_margin_risk: float = Field(..., ge=0.0)
    excess_capital_locked: float = Field(..., ge=0.0)
    recommended_action: str


class BatchScoreResponse(BaseModel):
    count: int = Field(..., description="Number of successfully matched SKUs")
    results: List[BatchScoreItem] = Field(..., description="Scored SKU results")
    unmatched_sku_ids: List[str] = Field(default_factory=list, description="Requested SKU IDs not found in catalog")
    data_disclaimer: str
