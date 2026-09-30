import pytest
import pandas as pd
from pathlib import Path

from dashboard.app import format_inr, load_all_artifacts


def test_format_inr_helper():
    assert format_inr(5000) == "₹5,000.00"
    assert format_inr(250000) == "₹2.50 L"
    assert format_inr(15000000) == "₹1.50 Cr"
    assert format_inr(0) == "₹0.00"
    assert format_inr(None) == "₹0.00"


def test_dashboard_artifacts_load_cleanly():
    risk_df, forecast_df, weekly_df, comp_df, qual_report = load_all_artifacts()
    assert risk_df is not None
    assert forecast_df is not None
    assert weekly_df is not None

    assert len(risk_df) == 300
    assert len(forecast_df) == 3600
    assert len(weekly_df) == 47400

    # Ensure required columns exist
    expected_risk_cols = [
        "sku_id", "product_name", "category", "current_on_hand", "forecast_weekly_demand",
        "lead_time_days", "safety_stock", "reorder_point", "weeks_of_cover",
        "stockout_risk_score", "overstock_risk_score", "risk_quadrant",
        "recommended_action", "lost_margin_risk", "excess_capital_locked", "priority"
    ]
    for col in expected_risk_cols:
        assert col in risk_df.columns


def test_dashboard_filtering_logic():
    risk_df, _, _, _, _ = load_all_artifacts()

    # Category filter
    appliances = risk_df[risk_df["category"] == "Small Appliances"]
    assert len(appliances) == 75

    # Quadrant filter
    reorders = risk_df[risk_df["risk_quadrant"] == "Reorder Now"]
    assert len(reorders) > 0

    # Empty filter state
    empty_slice = risk_df[(risk_df["category"] == "Small Appliances") & (risk_df["category"] == "Home Furnishings")]
    assert len(empty_slice) == 0
