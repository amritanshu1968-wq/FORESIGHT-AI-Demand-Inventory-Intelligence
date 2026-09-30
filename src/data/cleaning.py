"""
PROJECT FORESIGHT — Data Cleaning Pipeline
Standardizes data types, validates structural cleanliness, and logs all cleaning decisions.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from config import PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("data_cleaning")


def clean_datasets(
    sku_master: pd.DataFrame,
    calendar: pd.DataFrame,
    sales_daily: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
    save_processed: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleans, type-casts, and standardizes all four datasets.
    Logs each audit decision and creates an actionable audit trail.
    """
    logger.info("Executing reproducible data cleaning and standardization...")
    audit_records: List[Dict[str, str]] = []

    # 1. Clean sku_master
    sku_clean = sku_master.copy()
    init_sku_rows = len(sku_clean)
    sku_clean = sku_clean.drop_duplicates(subset=["sku_id"])
    sku_dups = init_sku_rows - len(sku_clean)
    if sku_dups > 0:
        audit_records.append({
            "dataset": "sku_master",
            "issue": "Duplicate SKU keys",
            "rows_affected": str(sku_dups),
            "action_taken": "Deduplicated by sku_id",
            "reason": "Primary key uniqueness",
        })

    # Standardize string fields
    sku_clean["category"] = sku_clean["category"].astype(str).str.strip()
    sku_clean["subcategory"] = sku_clean["subcategory"].astype(str).str.strip()
    sku_clean["brand"] = sku_clean["brand"].astype(str).str.strip()

    # 2. Clean calendar
    cal_clean = calendar.copy()
    init_cal_rows = len(cal_clean)
    cal_clean = cal_clean.drop_duplicates(subset=["date"])
    cal_dups = init_cal_rows - len(cal_clean)
    if cal_dups > 0:
        audit_records.append({
            "dataset": "calendar",
            "issue": "Duplicate calendar dates",
            "rows_affected": str(cal_dups),
            "action_taken": "Deduplicated by date",
            "reason": "Date dimension primary key uniqueness",
        })

    cal_clean["date"] = pd.to_datetime(cal_clean["date"]).dt.strftime("%Y-%m-%d")

    # 3. Clean sales_daily
    sales_clean = sales_daily.copy()
    init_sales_rows = len(sales_clean)
    sales_clean = sales_clean.drop_duplicates(subset=["date", "sku_id"])
    sales_dups = init_sales_rows - len(sales_clean)
    if sales_dups > 0:
        audit_records.append({
            "dataset": "sales_daily",
            "issue": "Duplicate (date, sku_id) transactions",
            "rows_affected": str(sales_dups),
            "action_taken": "Deduplicated composite key",
            "reason": "Transaction fact uniqueness",
        })

    # Ensure date format and types
    sales_clean["date"] = pd.to_datetime(sales_clean["date"]).dt.strftime("%Y-%m-%d")
    sales_clean["units_sold"] = sales_clean["units_sold"].astype(int)
    sales_clean["unit_price"] = sales_clean["unit_price"].astype(float).round(2)
    sales_clean["revenue"] = sales_clean["revenue"].astype(float).round(2)
    sales_clean["promotion_flag"] = sales_clean["promotion_flag"].astype(int)
    sales_clean["discount_pct"] = sales_clean["discount_pct"].astype(float).round(4)
    sales_clean["stockout_flag"] = sales_clean["stockout_flag"].astype(int)

    # 4. Clean inventory_snapshots
    inv_clean = inventory_snapshots.copy()
    init_inv_rows = len(inv_clean)
    inv_clean = inv_clean.drop_duplicates(subset=["snapshot_date", "sku_id"])
    inv_dups = init_inv_rows - len(inv_clean)
    if inv_dups > 0:
        audit_records.append({
            "dataset": "inventory_snapshots",
            "issue": "Duplicate inventory snapshots",
            "rows_affected": str(inv_dups),
            "action_taken": "Deduplicated composite key",
            "reason": "Inventory snapshot uniqueness",
        })

    inv_clean["snapshot_date"] = pd.to_datetime(inv_clean["snapshot_date"]).dt.strftime("%Y-%m-%d")
    inv_clean["opening_stock"] = inv_clean["opening_stock"].astype(int)
    inv_clean["stock_received"] = inv_clean["stock_received"].astype(int)
    inv_clean["units_sold"] = inv_clean["units_sold"].astype(int)
    inv_clean["closing_stock"] = inv_clean["closing_stock"].astype(int)
    inv_clean["on_order_units"] = inv_clean["on_order_units"].astype(int)
    inv_clean["days_of_cover"] = inv_clean["days_of_cover"].astype(float).round(1)
    inv_clean["inventory_value"] = inv_clean["inventory_value"].astype(float).round(2)

    # If all datasets passed without requiring row removal/alteration
    if not audit_records:
        audit_records.append({
            "dataset": "all_datasets",
            "issue": "None",
            "rows_affected": "0",
            "action_taken": "Schema verified, types standardized, 0 rows modified",
            "reason": "No corrective cleaning required; validation passed.",
        })
        logger.info("Cleaning Audit: No corrective cleaning required; validation passed.")

    audit_df = pd.DataFrame(audit_records)

    if save_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        sku_clean.to_csv(PROCESSED_DATA_DIR / "clean_sku_master.csv", index=False)
        cal_clean.to_csv(PROCESSED_DATA_DIR / "clean_calendar.csv", index=False)
        sales_clean.to_csv(PROCESSED_DATA_DIR / "clean_sales_daily.csv", index=False)
        inv_clean.to_csv(PROCESSED_DATA_DIR / "clean_inventory_snapshots.csv", index=False)
        audit_df.to_csv(PROCESSED_DATA_DIR / "cleaning_audit_log.csv", index=False)
        logger.info("Cleaned datasets and cleaning_audit_log.csv saved to %s", PROCESSED_DATA_DIR)

    return sku_clean, cal_clean, sales_clean, inv_clean, audit_df


if __name__ == "__main__":
    sku_df = pd.read_csv(SYNTHETIC_DATA_DIR / "sku_master.csv")
    cal_df = pd.read_csv(SYNTHETIC_DATA_DIR / "calendar.csv")
    sales_df = pd.read_csv(SYNTHETIC_DATA_DIR / "sales_daily.csv")
    inv_df = pd.read_csv(SYNTHETIC_DATA_DIR / "inventory_snapshots.csv")
    clean_datasets(sku_df, cal_df, sales_df, inv_df)
