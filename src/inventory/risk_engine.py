"""
PROJECT FORESIGHT — 2D Inventory Risk Engine
Computes safety stock, reorder point, weeks of cover, normalized risk scores,
and assigns every SKU to exactly one quadrant on the 2D Decision Grid.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from config import DEFAULT_SERVICE_LEVEL, OVERSTOCK_COVER_WEEKS_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger("risk_engine")


def get_z_score(service_level: float = DEFAULT_SERVICE_LEVEL) -> float:
    """Returns standard normal Z-score for target service level."""
    z_map = {
        0.90: 1.2816,
        0.95: 1.6449,
        0.98: 2.0537,
        0.99: 2.3263,
    }
    # Closest match
    return min(z_map.items(), key=lambda x: abs(x[0] - service_level))[1]


def calculate_safety_stock(
    lead_time_days: float,
    sigma_weekly_error: float,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> int:
    """
    Safety Stock = ceil(Z * sigma_error * sqrt(lead_time_weeks))
    Ensures consistent weekly unit convention.
    """
    lead_time_weeks = max(lead_time_days / 7.0, 0.5)
    z = get_z_score(service_level)
    ss = int(np.ceil(z * max(sigma_weekly_error, 0.5) * np.sqrt(lead_time_weeks)))
    return max(ss, 1)


def calculate_reorder_point(
    expected_lead_time_demand: float,
    safety_stock: int,
) -> int:
    """Reorder Point (ROP) = ceil(Expected Lead-Time Demand + Safety Stock)."""
    return int(np.ceil(expected_lead_time_demand + safety_stock))


def calculate_weeks_of_cover(
    current_on_hand: int,
    forecast_weekly_demand: float,
) -> float:
    """
    Weeks of Cover = Current On-Hand / Forecast Weekly Demand.
    Handles zero-demand SKUs gracefully (capped at 52 weeks).
    """
    if forecast_weekly_demand <= 0.05:
        return 52.0 if current_on_hand > 0 else 0.0
    return round(float(current_on_hand / forecast_weekly_demand), 1)


def calculate_stockout_risk_score(
    effective_available: float,
    lead_time_demand: float,
    reorder_point: float,
    safety_stock: float,
    demand_cv: float = 0.0,
) -> float:
    """
    Computes a normalized stockout risk score in [0.0, 1.0].
    Reflects the probability and depth of inventory exhaustion before replenishment arrives.
    High demand variability (CV) increases tail stockout risk.
    """
    if effective_available <= 0:
        return 1.0

    ltd = max(lead_time_demand, 1.0)
    ss = max(safety_stock, 1.0)

    if effective_available < ltd:
        base_score = 1.0 - 0.5 * (effective_available / ltd)
    elif effective_available < reorder_point:
        ratio = (effective_available - ltd) / ss
        base_score = 0.5 * (1.0 - ratio)
    else:
        buffer = (effective_available - reorder_point) / ltd
        base_score = max(0.0, 0.20 * (1.0 - min(buffer, 1.0)))

    # If demand is highly volatile / erratic (CV >= 0.85), tail risk of surge stockout is elevated
    if demand_cv >= 0.85:
        base_score = max(base_score, 0.52 + min(0.35, (demand_cv - 0.85) * 0.40))
    else:
        base_score += min(0.18, max(0.0, demand_cv) * 0.18)

    return round(float(np.clip(base_score, 0.0, 1.0)), 4)


def calculate_overstock_risk_score(
    weeks_of_cover: float,
    lead_time_weeks: float,
    overstock_threshold_weeks: float = OVERSTOCK_COVER_WEEKS_THRESHOLD,
    demand_cv: float = 0.0,
) -> float:
    """
    Computes a normalized overstock risk score in [0.0, 1.0].
    Evaluates excess forward coverage beyond target operating cycle.
    For highly volatile items (CV >= 0.85) holding stock, risk of demand collapse is high.
    """
    target_cover = lead_time_weeks + 4.0  # Safe buffer = lead time + 4 weeks of cycle stock

    if weeks_of_cover <= target_cover:
        base_score = 0.0
    elif weeks_of_cover <= overstock_threshold_weeks:
        ratio = (weeks_of_cover - target_cover) / max(overstock_threshold_weeks - target_cover, 1.0)
        base_score = 0.50 * ratio
    else:
        excess = (weeks_of_cover - overstock_threshold_weeks) / 15.0
        base_score = 0.50 + 0.50 * min(excess, 1.0)

    # If demand is highly volatile / erratic (CV >= 0.85) and has inventory on hand (WOC >= 1.5),
    # risk of demand sudden drop leaving stranded stock is elevated
    if demand_cv >= 0.85 and weeks_of_cover >= 1.5:
        base_score = max(base_score, 0.52 + min(0.35, (demand_cv - 0.85) * 0.40))
    elif weeks_of_cover > 3.0:
        base_score += min(0.18, max(0.0, demand_cv) * 0.18)

    return round(float(np.clip(base_score, 0.0, 1.0)), 4)


def assign_risk_quadrant(
    stockout_risk_score: float,
    overstock_risk_score: float,
    stockout_threshold: float = 0.50,
    overstock_threshold: float = 0.50,
) -> str:
    """
    Assigns SKU to exactly one of the 4 decision grid quadrants:
    - High Stockout, Low Overstock  -> 'Reorder Now'
    - Low Stockout, High Overstock  -> 'Markdown / Clear'
    - High Stockout, High Overstock -> 'Watch / Volatile'
    - Low Stockout, Low Overstock   -> 'Healthy'
    """
    is_high_stockout = stockout_risk_score >= stockout_threshold
    is_high_overstock = overstock_risk_score >= overstock_threshold

    if is_high_stockout and not is_high_overstock:
        return "Reorder Now"
    elif not is_high_stockout and is_high_overstock:
        return "Markdown / Clear"
    elif is_high_stockout and is_high_overstock:
        return "Watch / Volatile"
    else:
        return "Healthy"
