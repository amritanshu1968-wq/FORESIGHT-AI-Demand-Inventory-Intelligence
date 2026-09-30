"""
PROJECT FORESIGHT — Rupee (₹) Business Impact Quantification
Calculates lost margin at risk, excess working capital locked, inventory values, and priorities.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def calculate_rupee_impact(
    effective_available: int,
    current_on_hand: int,
    lead_time_demand: float,
    target_stock: int,
    unit_price: float,
    unit_cost: float,
) -> Tuple[float, float, float, float]:
    """
    Computes rigorous financial impact metrics in INR:
    1. Lost Margin Risk = max(0, lead_time_demand - effective_available) * (unit_price - unit_cost)
    2. Excess Capital Locked = max(0, current_on_hand - target_stock) * unit_cost
    3. Current Inventory Value = current_on_hand * unit_cost
    4. Total Financial Exposure = Lost Margin Risk + Excess Capital Locked
    """
    margin_per_unit = max(0.0, unit_price - unit_cost)

    # 1. Lost Margin Risk (Stockout vulnerability)
    lost_units = max(0.0, lead_time_demand - effective_available)
    lost_margin_risk = round(float(lost_units * margin_per_unit), 2)

    # 2. Excess Capital Locked (Overstock vulnerability)
    excess_units = max(0.0, float(current_on_hand - target_stock))
    excess_capital_locked = round(float(excess_units * unit_cost), 2)

    # 3. Inventory Value
    inventory_val = round(float(current_on_hand * unit_cost), 2)

    # 4. Total Financial Exposure
    total_exposure = round(lost_margin_risk + excess_capital_locked, 2)

    return lost_margin_risk, excess_capital_locked, inventory_val, total_exposure


def assign_action_priority(
    quadrant: str,
    total_exposure: float,
    high_threshold: float,
    medium_threshold: float,
) -> str:
    """
    Assigns transparent operational priority based on decision quadrant and rupee impact.
    HIGH: Actionable quadrant + substantial financial exposure.
    MEDIUM: Moderate exposure or watch list.
    LOW: Healthy inventory position or minimal rupee value.
    """
    if quadrant in ["Reorder Now", "Markdown / Clear"]:
        if total_exposure >= high_threshold:
            return "HIGH"
        elif total_exposure >= medium_threshold:
            return "MEDIUM"
        else:
            return "LOW"
    elif quadrant == "Watch / Volatile":
        return "HIGH" if total_exposure >= medium_threshold else "MEDIUM"
    else:  # Healthy
        return "LOW"
