"""
PROJECT FORESIGHT — Data Quality Profiler
Performs rigorous structural, statistical, and relational profiling of all four datasets.
"""

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd

from config import REPORTS_DIR, SYNTHETIC_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("quality_report")


def profile_dataframe(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Generates comprehensive column-by-column statistics for a dataframe."""
    profile: Dict[str, Any] = {
        "name": name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {},
    }

    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        col_info: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing_count": n_missing,
            "missing_pct": round(n_missing / len(df) * 100, 2),
            "unique_count": int(series.nunique(dropna=False)),
        }

        # Numeric statistics
        if pd.api.types.is_numeric_dtype(series):
            clean_s = series.dropna()
            if len(clean_s) > 0:
                col_info.update({
                    "min": round(float(clean_s.min()), 2),
                    "max": round(float(clean_s.max()), 2),
                    "mean": round(float(clean_s.mean()), 2),
                    "median": round(float(clean_s.median()), 2),
                    "std": round(float(clean_s.std()), 2) if len(clean_s) > 1 else 0.0,
                    "negative_count": int((clean_s < 0).sum()),
                })
        else:
            # Top categories/values
            top_vals = series.value_counts(dropna=False).head(3).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in top_vals.items()}

        profile["columns"][col] = col_info

    return profile


def run_full_quality_profile(
    sku_master: pd.DataFrame,
    calendar: pd.DataFrame,
    sales_daily: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
) -> Dict[str, Any]:
    """Runs quality profiling across all four datasets and compiles an audit report."""
    logger.info("Starting comprehensive data quality profiling across all 4 tables...")

    quality_report: Dict[str, Any] = {
        "title": "PROJECT FORESIGHT — Data Quality Profiling Report",
        "data_source": "Synthetic retail dataset generated for project development and demonstration",
        "tables": {},
    }

    # Table profiles
    quality_report["tables"]["sku_master"] = profile_dataframe(sku_master, "sku_master")
    quality_report["tables"]["calendar"] = profile_dataframe(calendar, "calendar")
    quality_report["tables"]["sales_daily"] = profile_dataframe(sales_daily, "sales_daily")
    quality_report["tables"]["inventory_snapshots"] = profile_dataframe(inventory_snapshots, "inventory_snapshots")

    # Relational & Business Integrity Checks
    logger.info("Profiling cross-table integrity, consistency, and outliers...")
    sku_set = set(sku_master["sku_id"])
    date_set = set(calendar["date"])

    # Revenue arithmetic
    rev_calc = np.round(sales_daily["units_sold"] * sales_daily["unit_price"], 2)
    rev_diff = np.abs(sales_daily["revenue"] - rev_calc)
    rev_inconsistencies = int((rev_diff > 0.05).sum())

    # Margin arithmetic
    margin_calc = np.round(sku_master["unit_price"] - sku_master["unit_cost"], 2)
    margin_diff = np.abs(sku_master["margin_per_unit"] - margin_calc)
    margin_inconsistencies = int((margin_diff > 0.05).sum())

    # Inventory conservation
    expected_closing = (
        inventory_snapshots["opening_stock"]
        + inventory_snapshots["stock_received"]
        - inventory_snapshots["units_sold"]
    )
    inv_discrepancies = int((inventory_snapshots["closing_stock"] != expected_closing).sum())

    quality_report["integrity_checks"] = {
        "orphan_skus_sales": int(len(set(sales_daily["sku_id"]) - sku_set)),
        "orphan_skus_inventory": int(len(set(inventory_snapshots["sku_id"]) - sku_set)),
        "orphan_dates_sales": int(len(set(sales_daily["date"]) - date_set)),
        "orphan_dates_inventory": int(len(set(inventory_snapshots["snapshot_date"]) - date_set)),
        "revenue_arithmetic_errors": rev_inconsistencies,
        "margin_arithmetic_errors": margin_inconsistencies,
        "inventory_balance_violations": inv_discrepancies,
        "negative_sales_records": int((sales_daily["units_sold"] < 0).sum()),
        "negative_inventory_records": int((inventory_snapshots["closing_stock"] < 0).sum()),
    }

    # Save to outputs/reports/
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "data_quality_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)
    logger.info("Data quality JSON report saved to %s", json_path)

    return quality_report


if __name__ == "__main__":
    sku_df = pd.read_csv(SYNTHETIC_DATA_DIR / "sku_master.csv")
    cal_df = pd.read_csv(SYNTHETIC_DATA_DIR / "calendar.csv")
    sales_df = pd.read_csv(SYNTHETIC_DATA_DIR / "sales_daily.csv")
    inv_df = pd.read_csv(SYNTHETIC_DATA_DIR / "inventory_snapshots.csv")
    run_full_quality_profile(sku_df, cal_df, sales_df, inv_df)
