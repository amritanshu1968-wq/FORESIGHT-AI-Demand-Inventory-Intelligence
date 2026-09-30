"""
PROJECT FORESIGHT — Weekly Aggregation Pipeline
Transforms daily SKU-level transactions into a weekly demand dataset for forecasting.

Grain: SKU + Week (Monday to Sunday)
Output: data/processed/weekly_demand.csv
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("aggregation")


def aggregate_daily_to_weekly(
    sales_daily: pd.DataFrame,
    sku_master: pd.DataFrame,
    calendar: pd.DataFrame,
    save_csv: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Aggregates daily sales to SKU-week grain.
    Ensures strict arithmetic reconciliation:
    total weekly units == total daily units sold
    total weekly revenue == total daily revenue
    """
    logger.info("Aggregating daily transactions into weekly SKU demand...")

    df_sales = sales_daily.copy()
    df_sales["date"] = pd.to_datetime(df_sales["date"])

    # Determine calendar week start (Monday) and week end (Sunday)
    # Day of week in pandas: Monday=0, Sunday=6
    df_sales["week_start"] = df_sales["date"] - pd.to_timedelta(df_sales["date"].dt.dayofweek, unit="D")
    df_sales["week_end"] = df_sales["week_start"] + pd.Timedelta(days=6)

    # Convert dates to ISO week and year
    iso_cal = df_sales["week_start"].dt.isocalendar()
    df_sales["year"] = iso_cal.year
    df_sales["week"] = iso_cal.week

    # Aggregate by (sku_id, week_start, week_end, year, week)
    agg_funcs = {
        "units_sold": "sum",
        "revenue": "sum",
        "promotion_flag": "sum",  # promotion_days in the week (0-7)
        "discount_pct": "mean",   # average discount applied
        "stockout_flag": "sum",   # stockout_days in the week (0-7)
    }

    weekly = (
        df_sales.groupby(
            ["sku_id", "week_start", "week_end", "year", "week"],
            as_index=False,
        )
        .agg(agg_funcs)
        .rename(
            columns={
                "units_sold": "weekly_units_sold",
                "revenue": "weekly_revenue",
                "promotion_flag": "promotion_days",
                "discount_pct": "average_discount",
                "stockout_flag": "stockout_days",
            }
        )
    )

    # Derived weekly metrics
    weekly["average_price"] = np.where(
        weekly["weekly_units_sold"] > 0,
        (weekly["weekly_revenue"] / weekly["weekly_units_sold"]).round(2),
        0.0,
    )
    weekly["average_discount"] = weekly["average_discount"].round(4)
    weekly["stockout_rate"] = (weekly["stockout_days"] / 7.0).round(4)
    weekly["is_promotional_week"] = (weekly["promotion_days"] >= 2).astype(int)

    # Join SKU dimension metadata
    sku_meta_cols = [
        "sku_id",
        "product_name",
        "category",
        "subcategory",
        "brand",
        "unit_price",
        "unit_cost",
        "margin_per_unit",
        "lead_time_days",
        "target_service_level",
        "status",
        "demand_profile",
    ]
    # Filter cols present in sku_master
    sku_meta_cols = [c for c in sku_meta_cols if c in sku_master.columns]
    weekly = pd.merge(weekly, sku_master[sku_meta_cols], on="sku_id", how="left")

    # Format dates
    weekly["week_start"] = pd.to_datetime(weekly["week_start"]).dt.strftime("%Y-%m-%d")
    weekly["week_end"] = pd.to_datetime(weekly["week_end"]).dt.strftime("%Y-%m-%d")

    # Sort deterministically
    weekly = weekly.sort_values(["sku_id", "week_start"]).reset_index(drop=True)

    # Reconciliation Audit
    total_daily_units = int(sales_daily["units_sold"].sum())
    total_weekly_units = int(weekly["weekly_units_sold"].sum())
    total_daily_rev = round(float(sales_daily["revenue"].sum()), 2)
    total_weekly_rev = round(float(weekly["weekly_revenue"].sum()), 2)

    units_diff = abs(total_daily_units - total_weekly_units)
    rev_diff = abs(total_daily_rev - total_weekly_rev)

    reconciliation = {
        "total_daily_units": total_daily_units,
        "total_weekly_units": total_weekly_units,
        "units_discrepancy": units_diff,
        "total_daily_revenue": total_daily_rev,
        "total_weekly_revenue": total_weekly_rev,
        "revenue_discrepancy": rev_diff,
        "total_sku_weeks": len(weekly),
        "total_skus": weekly["sku_id"].nunique(),
        "total_weeks": weekly["week_start"].nunique(),
        "start_week": str(weekly["week_start"].min()),
        "end_week": str(weekly["week_start"].max()),
    }

    if units_diff > 0 or rev_diff > 1.0:
        raise ValueError(
            f"Weekly aggregation reconciliation failed! Units diff: {units_diff}, Revenue diff: {rev_diff}"
        )

    logger.info(
        "Reconciliation SUCCESS: %d units sold and INR %.2f revenue match daily source exactly across %d SKU-weeks.",
        total_weekly_units,
        total_weekly_rev,
        len(weekly),
    )

    if save_csv:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DATA_DIR / "weekly_demand.csv"
        weekly.to_csv(out_path, index=False)
        logger.info("Saved weekly demand dataset to %s", out_path)

    return weekly, reconciliation


if __name__ == "__main__":
    sales_df = pd.read_csv(PROCESSED_DATA_DIR / "clean_sales_daily.csv")
    sku_df = pd.read_csv(PROCESSED_DATA_DIR / "clean_sku_master.csv")
    cal_df = pd.read_csv(PROCESSED_DATA_DIR / "clean_calendar.csv")
    aggregate_daily_to_weekly(sales_df, sku_df, cal_df)
