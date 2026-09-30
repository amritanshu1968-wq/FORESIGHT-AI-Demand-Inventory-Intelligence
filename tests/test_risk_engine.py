import pytest
import numpy as np
import pandas as pd

from src.inventory.risk_engine import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_weeks_of_cover,
    calculate_stockout_risk_score,
    calculate_overstock_risk_score,
    assign_risk_quadrant,
)
from src.inventory.recommendations import calculate_eoq, generate_sku_recommendation
from src.inventory.impact import calculate_rupee_impact


def test_safety_stock_calculation():
    # lead_time = 14 days = 2 weeks. sigma = 5.0. service level 0.95 (Z ~ 1.645)
    # SS = ceil(1.645 * 5.0 * sqrt(2)) = ceil(1.645 * 5 * 1.4142) = ceil(11.63) = 12
    ss = calculate_safety_stock(lead_time_days=14.0, sigma_weekly_error=5.0, service_level=0.95)
    assert ss == 12


def test_reorder_point():
    rop = calculate_reorder_point(expected_lead_time_demand=35.4, safety_stock=12)
    assert rop == 48  # ceil(35.4 + 12) = 48


def test_weeks_of_cover_safe_zero():
    # Regular case
    woc = calculate_weeks_of_cover(current_on_hand=100, forecast_weekly_demand=20.0)
    assert woc == 5.0

    # Zero demand case (must not raise ZeroDivisionError)
    woc_zero = calculate_weeks_of_cover(current_on_hand=50, forecast_weekly_demand=0.0)
    assert woc_zero == 52.0

    # Zero stock and zero demand
    woc_empty = calculate_weeks_of_cover(current_on_hand=0, forecast_weekly_demand=0.0)
    assert woc_empty == 0.0


def test_risk_score_bounds():
    for avail in [0, 5, 20, 50, 100, 500]:
        s_score = calculate_stockout_risk_score(
            effective_available=avail, lead_time_demand=30, reorder_point=45, safety_stock=15
        )
        assert 0.0 <= s_score <= 1.0

    for woc in [0.0, 2.0, 6.0, 10.0, 15.0, 52.0]:
        o_score = calculate_overstock_risk_score(weeks_of_cover=woc, lead_time_weeks=2.0)
        assert 0.0 <= o_score <= 1.0


def test_decision_grid_quadrants():
    assert assign_risk_quadrant(0.85, 0.15) == "Reorder Now"
    assert assign_risk_quadrant(0.15, 0.85) == "Markdown / Clear"
    assert assign_risk_quadrant(0.85, 0.85) == "Watch / Volatile"
    assert assign_risk_quadrant(0.15, 0.15) == "Healthy"


def test_rupee_impact_non_negativity_and_margin():
    lost_margin, excess_cap, inv_val, exposure = calculate_rupee_impact(
        effective_available=10,
        current_on_hand=10,
        lead_time_demand=30.0,  # deficit of 20 units
        target_stock=40,
        unit_price=1000.0,
        unit_cost=600.0,       # margin = 400.0
    )
    # Lost Margin = 20 * 400 = 8000.0 (NOT 20 * 1000 = 20000)
    assert lost_margin == 8000.0
    assert excess_cap == 0.0
    assert inv_val == 6000.0
    assert exposure == 8000.0


def test_eoq_calculation():
    # Annual demand = 520, cost = 100, S = 500, H = 100 * 0.2 = 20
    # EOQ = sqrt(2 * 520 * 500 / 20) = sqrt(26000) ~ 161.2 -> 162
    eoq = calculate_eoq(annual_demand=520, unit_cost=100.0, order_cost=500.0, moq=25)
    assert eoq == 162
