"""
PROJECT FORESIGHT — Data Validation Engine
Validates schema integrity, data types, relational constraints, and inventory conservation.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data_validation")


class DataValidationError(Exception):
    """Raised when critical data validation checks fail."""
    pass


def validate_primary_keys(
    sku_master: pd.DataFrame,
    calendar: pd.DataFrame,
    sales_daily: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
) -> Dict[str, Any]:
    """Validates uniqueness of primary/composite keys."""
    results = {}

    # 1. sku_master: sku_id
    sku_dup = sku_master["sku_id"].duplicated().sum()
    results["sku_master_pk_duplicates"] = int(sku_dup)
    if sku_dup > 0:
        raise DataValidationError(f"sku_master has {sku_dup} duplicate sku_id primary keys.")

    # 2. calendar: date
    cal_dup = calendar["date"].duplicated().sum()
    results["calendar_pk_duplicates"] = int(cal_dup)
    if cal_dup > 0:
        raise DataValidationError(f"calendar has {cal_dup} duplicate date primary keys.")

    # 3. sales_daily: (date, sku_id)
    sales_dup = sales_daily.duplicated(subset=["date", "sku_id"]).sum()
    results["sales_daily_composite_key_duplicates"] = int(sales_dup)
    if sales_dup > 0:
        raise DataValidationError(f"sales_daily has {sales_dup} duplicate (date, sku_id) records.")

    # 4. inventory_snapshots: (snapshot_date, sku_id)
    inv_dup = inventory_snapshots.duplicated(subset=["snapshot_date", "sku_id"]).sum()
    results["inventory_composite_key_duplicates"] = int(inv_dup)
    if inv_dup > 0:
        raise DataValidationError(f"inventory_snapshots has {inv_dup} duplicate (snapshot_date, sku_id) records.")

    return results


def validate_foreign_keys(
    sku_master: pd.DataFrame,
    calendar: pd.DataFrame,
    sales_daily: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
) -> Dict[str, Any]:
    """Validates referential integrity across dimension and fact tables."""
    results = {}
    valid_skus = set(sku_master["sku_id"])
    valid_dates = set(calendar["date"])

    # sales_daily -> sku_master
    orphan_sales_skus = set(sales_daily["sku_id"]) - valid_skus
    results["orphan_sales_skus_count"] = len(orphan_sales_skus)
    if orphan_sales_skus:
        raise DataValidationError(f"sales_daily contains {len(orphan_sales_skus)} orphan SKU references.")

    # sales_daily -> calendar
    orphan_sales_dates = set(sales_daily["date"]) - valid_dates
    results["orphan_sales_dates_count"] = len(orphan_sales_dates)
    if orphan_sales_dates:
        raise DataValidationError(f"sales_daily contains {len(orphan_sales_dates)} orphan date references.")

    # inventory_snapshots -> sku_master
    orphan_inv_skus = set(inventory_snapshots["sku_id"]) - valid_skus
    results["orphan_inventory_skus_count"] = len(orphan_inv_skus)
    if orphan_inv_skus:
        raise DataValidationError(f"inventory_snapshots contains {len(orphan_inv_skus)} orphan SKU references.")

    # inventory_snapshots -> calendar
    orphan_inv_dates = set(inventory_snapshots["snapshot_date"]) - valid_dates
    results["orphan_inventory_dates_count"] = len(orphan_inv_dates)
    if orphan_inv_dates:
        raise DataValidationError(f"inventory_snapshots contains {len(orphan_inv_dates)} orphan date references.")

    return results


def validate_non_negativity_and_ranges(
    sku_master: pd.DataFrame,
    sales_daily: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
) -> Dict[str, Any]:
    """Checks for negative or economically impossible values."""
    results = {}

    # SKU master checks
    neg_costs = (sku_master["unit_cost"] <= 0).sum()
    neg_prices = (sku_master["unit_price"] <= 0).sum()
    cost_exceeds_price = (sku_master["unit_cost"] >= sku_master["unit_price"]).sum()
    margin_mismatch = (np.abs((sku_master["unit_price"] - sku_master["unit_cost"]) - sku_master["margin_per_unit"]) > 0.02).sum()

    results["invalid_costs"] = int(neg_costs)
    results["invalid_prices"] = int(neg_prices)
    results["cost_exceeds_price"] = int(cost_exceeds_price)
    results["margin_arithmetic_errors"] = int(margin_mismatch)

    if neg_costs or neg_prices or cost_exceeds_price or margin_mismatch:
        raise DataValidationError("sku_master contains invalid financial or price/cost relationships.")

    # Sales daily checks
    neg_sales = (sales_daily["units_sold"] < 0).sum()
    neg_rev = (sales_daily["revenue"] < 0).sum()
    invalid_discount = ((sales_daily["discount_pct"] < 0) | (sales_daily["discount_pct"] > 1.0)).sum()

    results["negative_units_sold"] = int(neg_sales)
    results["negative_revenue"] = int(neg_rev)
    results["invalid_discount_pct"] = int(invalid_discount)

    if neg_sales or neg_rev or invalid_discount:
        raise DataValidationError("sales_daily contains negative units, negative revenue, or invalid discount percentages.")

    # Inventory snapshots checks
    neg_open = (inventory_snapshots["opening_stock"] < 0).sum()
    neg_recv = (inventory_snapshots["stock_received"] < 0).sum()
    neg_close = (inventory_snapshots["closing_stock"] < 0).sum()

    results["negative_opening_stock"] = int(neg_open)
    results["negative_stock_received"] = int(neg_recv)
    results["negative_closing_stock"] = int(neg_close)

    if neg_open or neg_recv or neg_close:
        raise DataValidationError("inventory_snapshots contains negative stock quantities.")

    return results


def validate_inventory_conservation(
    sales_daily: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
) -> Dict[str, Any]:
    """
    CRITICAL CHECK:
    1. Closing(t) == Opening(t) + Stock_Received(t) - Units_Sold(t) on every row.
    2. Physical sales limit: Units_Sold(t) <= Opening(t) + Stock_Received(t).
    3. Inter-temporal continuity: Opening(t+1) == Closing(t) per SKU.
    4. Exact match of units_sold between sales_daily and inventory_snapshots.
    """
    results = {}

    # Check 1: Row-level balance equation
    expected_closing = (
        inventory_snapshots["opening_stock"]
        + inventory_snapshots["stock_received"]
        - inventory_snapshots["units_sold"]
    )
    balance_discrepancies = (inventory_snapshots["closing_stock"] != expected_closing).sum()
    results["balance_equation_violations"] = int(balance_discrepancies)
    if balance_discrepancies > 0:
        raise DataValidationError(
            f"Found {balance_discrepancies} rows violating: Closing = Opening + Received - Sold."
        )

    # Check 2: Physical sales limit
    oversold = (
        inventory_snapshots["units_sold"]
        > (inventory_snapshots["opening_stock"] + inventory_snapshots["stock_received"])
    ).sum()
    results["oversold_violations"] = int(oversold)
    if oversold > 0:
        raise DataValidationError(f"Found {oversold} rows where units sold exceeded total available stock.")

    # Check 3: Cross-table sales consistency
    # Merge sales_daily with inventory_snapshots on (date, sku_id)
    merged = pd.merge(
        sales_daily[["date", "sku_id", "units_sold", "stockout_flag"]],
        inventory_snapshots[["snapshot_date", "sku_id", "units_sold", "stockout_flag"]],
        left_on=["date", "sku_id"],
        right_on=["snapshot_date", "sku_id"],
        suffixes=("_sales", "_inv"),
    )
    sales_mismatch = (merged["units_sold_sales"] != merged["units_sold_inv"]).sum()
    stockout_mismatch = (merged["stockout_flag_sales"] != merged["stockout_flag_inv"]).sum()

    results["cross_table_sales_mismatch"] = int(sales_mismatch)
    results["cross_table_stockout_mismatch"] = int(stockout_mismatch)

    if sales_mismatch > 0 or stockout_mismatch > 0:
        raise DataValidationError("Sales and inventory snapshot tables disagree on units sold or stockout flags.")

    # Check 4: Inter-temporal continuity: Opening(t+1) == Closing(t)
    sorted_inv = inventory_snapshots.sort_values(["sku_id", "snapshot_date"]).copy()
    sorted_inv["prev_closing"] = sorted_inv.groupby("sku_id")["closing_stock"].shift(1)
    # Ignore the first day of each SKU (where prev_closing is NaN)
    continuity_slice = sorted_inv.dropna(subset=["prev_closing"])
    continuity_breaks = (continuity_slice["opening_stock"] != continuity_slice["prev_closing"]).sum()
    results["temporal_continuity_breaks"] = int(continuity_breaks)

    if continuity_breaks > 0:
        raise DataValidationError(f"Found {continuity_breaks} inter-day stock continuity breaks: Opening(t+1) != Closing(t).")

    return results


def run_comprehensive_validation(
    sku_master: pd.DataFrame,
    calendar: pd.DataFrame,
    sales_daily: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
) -> Dict[str, Any]:
    """Runs all validation suites and compiles a comprehensive health report."""
    logger.info("Executing comprehensive relational and arithmetic data validation...")

    report: Dict[str, Any] = {}

    # Dataset dimensions
    report["dataset_summary"] = {
        "sku_master_rows": len(sku_master),
        "sku_master_cols": len(sku_master.columns),
        "calendar_rows": len(calendar),
        "calendar_cols": len(calendar.columns),
        "sales_daily_rows": len(sales_daily),
        "sales_daily_cols": len(sales_daily.columns),
        "inventory_snapshots_rows": len(inventory_snapshots),
        "inventory_snapshots_cols": len(inventory_snapshots.columns),
    }

    # Date ranges
    report["date_range"] = {
        "start_date": str(sales_daily["date"].min()),
        "end_date": str(sales_daily["date"].max()),
        "total_days": calendar["date"].nunique(),
    }

    # SKU count and category breakdown
    report["sku_coverage"] = {
        "total_skus": sku_master["sku_id"].nunique(),
        "categories": sku_master["category"].value_counts().to_dict(),
    }

    # Missing values
    report["missing_values"] = {
        "sku_master": int(sku_master.isna().sum().sum()),
        "calendar": int(calendar.drop(columns=["holiday_name", "festival_name", "promo_event"], errors="ignore").isna().sum().sum()),
        "sales_daily": int(sales_daily.isna().sum().sum()),
        "inventory_snapshots": int(inventory_snapshots.isna().sum().sum()),
    }

    # Duplicate rows
    report["duplicate_rows"] = {
        "sku_master": int(sku_master.duplicated().sum()),
        "calendar": int(calendar.duplicated().sum()),
        "sales_daily": int(sales_daily.duplicated().sum()),
        "inventory_snapshots": int(inventory_snapshots.duplicated().sum()),
    }

    # Execute assertions
    pk_results = validate_primary_keys(sku_master, calendar, sales_daily, inventory_snapshots)
    fk_results = validate_foreign_keys(sku_master, calendar, sales_daily, inventory_snapshots)
    range_results = validate_non_negativity_and_ranges(sku_master, sales_daily, inventory_snapshots)
    inv_results = validate_inventory_conservation(sales_daily, inventory_snapshots)

    report["validation_checks"] = {
        **pk_results,
        **fk_results,
        **range_results,
        **inv_results,
    }

    # Operational metrics
    total_sales_days = len(sales_daily)
    stockout_count = int(sales_daily["stockout_flag"].sum())
    report["operational_metrics"] = {
        "stockout_events_count": stockout_count,
        "stockout_percentage": round(stockout_count / total_sales_days * 100, 2),
        "overstock_snapshot_count": int((inventory_snapshots["days_of_cover"] > 70).sum()),
        "overstock_percentage": round(
            float((inventory_snapshots["days_of_cover"] > 70).sum() / len(inventory_snapshots) * 100), 2
        ),
        "total_units_sold": int(sales_daily["units_sold"].sum()),
        "total_revenue_inr": round(float(sales_daily["revenue"].sum()), 2),
    }

    logger.info("All validation suites passed successfully with 0 defects!")
    return report
