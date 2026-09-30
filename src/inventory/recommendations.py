"""
PROJECT FORESIGHT — Inventory Recommendations Engine
Calculates EOQ, MOQ-adjusted reorder batch sizes, excess units to clear, and actionable guidance.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from config import ANNUAL_HOLDING_COST_RATE


def calculate_eoq(
    annual_demand: float,
    unit_cost: float,
    order_cost: float = 500.0,
    holding_cost_rate: float = ANNUAL_HOLDING_COST_RATE,
    moq: int = 15,
) -> int:
    """
    Economic Order Quantity:
    EOQ = sqrt((2 * D * S) / H)
    where H = unit_cost * holding_cost_rate.
    Adjusted to be at least MOQ.
    """
    if annual_demand <= 0:
        return moq

    h = max(unit_cost * holding_cost_rate, 1.0)
    raw_eoq = np.sqrt((2.0 * annual_demand * order_cost) / h)
    eoq = int(np.ceil(raw_eoq))
    return max(moq, eoq)


def generate_sku_recommendation(
    quadrant: str,
    current_on_hand: int,
    effective_available: int,
    reorder_point: int,
    forecast_weekly_demand: float,
    lead_time_weeks: float,
    eoq: int,
    moq: int,
) -> Tuple[str, int, int]:
    """
    Generates deterministic operational recommendations:
    Returns (recommended_action, recommended_order_qty, excess_units).
    """
    target_stock = int(np.ceil(max(reorder_point, (lead_time_weeks + 4.0) * forecast_weekly_demand)))

    if quadrant == "Reorder Now":
        deficit = max(0, reorder_point - effective_available)
        order_qty = max(moq, eoq, deficit + eoq)
        excess_units = 0
        action = f"Urgent replenishment order of {order_qty} units needed to avoid stockout."

    elif quadrant == "Markdown / Clear":
        order_qty = 0
        excess_units = max(0, current_on_hand - target_stock)
        action = f"Promotional markdown recommended on {excess_units} excess units to recover working capital."

    elif quadrant == "Watch / Volatile":
        order_qty = 0
        excess_units = max(0, current_on_hand - target_stock) if current_on_hand > target_stock else 0
        action = "High demand volatility; review supplier lead time & manual audit before reordering."

    else:  # Healthy
        order_qty = 0
        excess_units = 0
        action = "Stock position healthy; maintain routine monitoring."

    return action, order_qty, excess_units
