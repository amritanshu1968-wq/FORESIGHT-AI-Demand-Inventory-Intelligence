"""
PROJECT FORESIGHT — Exploratory Data Analysis & Visualization
Computes demand patterns, seasonality, promotions, stockouts, ABC Pareto, and inventory health.
Generates 11 publication-grade charts saved in outputs/charts/.
"""

import json
from pathlib import Path
from typing import Any, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CHARTS_DIR, DATASET_DISCLAIMER, PROCESSED_DATA_DIR, REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger("eda_analysis")

# Visual Theme Constants
PRIMARY = "#2C3E50"    # Deep Navy
ACCENT = "#3498DB"     # Royal Blue
EMERALD = "#2ECC71"    # Green
AMBER = "#F39C12"      # Gold / Orange
CRIMSON = "#E74C3C"    # Coral / Red
PURPLE = "#8E44AD"     # Violet
GREY = "#95A5A6"       # Cool Slate

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": "#BDC3C7",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def generate_all_eda_charts(
    sales_daily: pd.DataFrame,
    sku_master: pd.DataFrame,
    calendar: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
    weekly_demand: pd.DataFrame,
) -> Dict[str, Any]:
    """Computes all statistical metrics and saves 11 EDA charts."""
    logger.info("Computing EDA metrics and generating 11 analytical charts...")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_summary: Dict[str, Any] = {}

    # -------------------------------------------------------------
    # 1. Overall Demand Trend
    # -------------------------------------------------------------
    weekly_tot = weekly_demand.groupby("week_start")["weekly_units_sold"].sum().reset_index()
    weekly_tot["week_start"] = pd.to_datetime(weekly_tot["week_start"])
    weekly_tot = weekly_tot.sort_values("week_start")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(weekly_tot["week_start"], weekly_tot["weekly_units_sold"], color=ACCENT, lw=2, label="Weekly Units Sold")
    # 12-week rolling average trend line
    rolling_trend = weekly_tot["weekly_units_sold"].rolling(12, min_periods=1, center=True).mean()
    ax.plot(weekly_tot["week_start"], rolling_trend, color=PRIMARY, lw=2.2, ls="--", label="12-Week Rolling Trend")
    ax.set_title("Overall Weekly Demand Trend (2023 – 2025)", pad=12)
    ax.set_xlabel("Timeline")
    ax.set_ylabel("Fulfillable Demand (Units Sold)")
    ax.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#E2E8F0")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "01_overall_demand_trend.png")
    plt.close(fig)

    metrics_summary["overall_demand"] = {
        "mean_weekly_units": round(float(weekly_tot["weekly_units_sold"].mean()), 1),
        "peak_weekly_units": int(weekly_tot["weekly_units_sold"].max()),
        "min_weekly_units": int(weekly_tot["weekly_units_sold"].min()),
    }

    # -------------------------------------------------------------
    # 2 & 3. Category Demand & Revenue Comparison
    # -------------------------------------------------------------
    cat_summary = (
        weekly_demand.groupby("category")
        .agg(total_units=("weekly_units_sold", "sum"), total_revenue=("weekly_revenue", "sum"))
        .sort_values("total_revenue", ascending=False)
    )
    total_rev = cat_summary["total_revenue"].sum()
    total_u = cat_summary["total_units"].sum()
    cat_summary["unit_share_pct"] = (cat_summary["total_units"] / total_u * 100).round(2)
    cat_summary["rev_share_pct"] = (cat_summary["total_revenue"] / total_rev * 100).round(2)

    # Chart 02: Category Demand
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(cat_summary.index, cat_summary["total_units"] / 1000, color=ACCENT)
    ax.set_title("Total Demand by Product Category (Thousands of Units)")
    ax.set_xlabel("Total Units Sold ('000)")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 3, bar.get_y() + bar.get_height() / 2, f"{w:.1f}k", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "02_category_demand.png")
    plt.close(fig)

    # Chart 03: Category Revenue
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(cat_summary.index, cat_summary["total_revenue"] / 1e7, color=EMERALD)
    ax.set_title("Total Revenue Contribution by Category (Crores ₹)")
    ax.set_xlabel("Revenue (₹ Crores)")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"₹{w:.1f} Cr", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "03_category_revenue.png")
    plt.close(fig)

    metrics_summary["category_breakdown"] = cat_summary.to_dict(orient="index")

    # -------------------------------------------------------------
    # 4. ABC / Pareto Volume & Revenue Analysis
    # -------------------------------------------------------------
    sku_agg = (
        weekly_demand.groupby("sku_id")
        .agg(units=("weekly_units_sold", "sum"), revenue=("weekly_revenue", "sum"))
        .sort_values("revenue", ascending=False)
    )
    sku_agg["cum_rev"] = sku_agg["revenue"].cumsum()
    sku_agg["cum_rev_pct"] = sku_agg["cum_rev"] / sku_agg["revenue"].sum() * 100.0

    # Dynamic Pareto Assignment: Top 70% = A, Next 20% (70-90%) = B, Last 10% = C
    sku_agg["abc_class"] = np.where(
        sku_agg["cum_rev_pct"] <= 70.0,
        "A",
        np.where(sku_agg["cum_rev_pct"] <= 90.0, "B", "C"),
    )
    # Ensure at least top 1 is A
    sku_agg.iloc[0, sku_agg.columns.get_loc("abc_class")] = "A"

    abc_counts = sku_agg["abc_class"].value_counts().to_dict()
    abc_rev_share = (sku_agg.groupby("abc_class")["revenue"].sum() / sku_agg["revenue"].sum() * 100).round(1).to_dict()

    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    colors = [{"A": PRIMARY, "B": AMBER, "C": GREY}[c] for c in sku_agg["abc_class"]]
    ax1.bar(range(len(sku_agg)), sku_agg["revenue"] / 1e6, color=colors, width=1.0)
    ax1.set_ylabel("Revenue per SKU (₹ Millions)")
    ax1.set_xlabel("SKUs Ranked by Revenue Contribution (1 to 300)")
    ax1.set_title(f"ABC Pareto Analysis: Class A ({abc_counts.get('A', 0)} SKUs) Drives {abc_rev_share.get('A', 0)}% of Revenue")

    ax2 = ax1.twinx()
    ax2.plot(range(len(sku_agg)), sku_agg["cum_rev_pct"], color=CRIMSON, lw=2)
    ax2.set_ylabel("Cumulative Revenue %")
    ax2.axhline(70, color=GREY, ls=":", alpha=0.7)
    ax2.axhline(90, color=GREY, ls=":", alpha=0.7)
    ax2.set_ylim(0, 105)
    ax2.spines["right"].set_visible(True)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "04_abc_pareto.png")
    plt.close(fig)

    metrics_summary["abc_pareto"] = {
        "counts": {str(k): int(v) for k, v in abc_counts.items()},
        "revenue_shares_pct": abc_rev_share,
        "class_a_sku_count": int(abc_counts.get("A", 0)),
        "class_a_sku_pct": round(int(abc_counts.get("A", 0)) / len(sku_agg) * 100, 1),
    }

    # -------------------------------------------------------------
    # 5. Weekly Seasonality (Day of Week Analysis)
    # -------------------------------------------------------------
    sales_cal = pd.merge(sales_daily, calendar[["date", "day_of_week"]], on="date", how="left")
    dow_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    dow_agg = sales_cal.groupby("day_of_week")["units_sold"].mean()
    dow_labels = [dow_names[i] for i in dow_agg.index]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar(dow_labels, dow_agg.values, color=[ACCENT if d < 6 else PRIMARY for d in dow_agg.index])
    ax.set_title("Average Daily Demand by Day of Week (Weekend Lift Pattern)")
    ax.set_ylabel("Average Units Sold per SKU-Day")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "05_weekly_seasonality.png")
    plt.close(fig)

    metrics_summary["weekly_seasonality"] = {
        "weekday_avg_units": round(float(dow_agg.loc[[1, 2, 3, 4, 5]].mean()), 2),
        "weekend_avg_units": round(float(dow_agg.loc[[6, 7]].mean()), 2),
        "weekend_lift_pct": round(
            float((dow_agg.loc[[6, 7]].mean() / dow_agg.loc[[1, 2, 3, 4, 5]].mean() - 1.0) * 100), 2
        ),
    }

    # -------------------------------------------------------------
    # 6. Monthly Seasonality
    # -------------------------------------------------------------
    sales_month = pd.merge(sales_daily, calendar[["date", "month"]], on="date", how="left")
    month_agg = sales_month.groupby("month")["units_sold"].mean()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(month_names, month_agg.values, marker="o", color=PRIMARY, lw=2.2, markersize=6)
    ax.fill_between(month_names, month_agg.values, color=ACCENT, alpha=0.15)
    ax.set_title("Monthly Demand Seasonality (Surge During Oct–Nov Festive Window)")
    ax.set_ylabel("Mean Units per SKU-Day")
    ax.set_xlabel("Calendar Month")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "06_monthly_seasonality.png")
    plt.close(fig)

    metrics_summary["monthly_seasonality"] = {
        "peak_month": month_names[int(month_agg.idxmax()) - 1],
        "peak_units": round(float(month_agg.max()), 2),
        "trough_month": month_names[int(month_agg.idxmin()) - 1],
        "trough_units": round(float(month_agg.min()), 2),
    }

    # -------------------------------------------------------------
    # 7. Festival Impact
    # -------------------------------------------------------------
    sales_fest = pd.merge(sales_daily, calendar[["date", "festival"]], on="date", how="left")
    fest_agg = sales_fest.groupby("festival")["units_sold"].mean()
    fest_lift_pct = round(float((fest_agg.loc[1] / fest_agg.loc[0] - 1.0) * 100), 2)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(["Non-Festival Days", "Festival Windows (Diwali/Holi)"], fest_agg.values, color=[GREY, CRIMSON], width=0.55)
    ax.set_title(f"Observed Festival Demand Surge (+{fest_lift_pct}% Lift)")
    ax.set_ylabel("Average Units Sold per SKU-Day")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.08, f"{h:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "07_festival_demand.png")
    plt.close(fig)

    metrics_summary["festival_lift"] = {
        "non_festival_avg": round(float(fest_agg.loc[0]), 2),
        "festival_avg": round(float(fest_agg.loc[1]), 2),
        "festival_lift_pct": fest_lift_pct,
    }

    # -------------------------------------------------------------
    # 8. Promotion Lift
    # -------------------------------------------------------------
    promo_agg = sales_daily.groupby("promotion_flag")["units_sold"].mean()
    promo_lift_pct = round(float((promo_agg.loc[1] / promo_agg.loc[0] - 1.0) * 100), 2)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(["Baseline (Non-Promo)", "Promotional Campaigns"], promo_agg.values, color=[GREY, AMBER], width=0.55)
    ax.set_title(f"Observed Promotional Demand Association (+{promo_lift_pct}% Lift)")
    ax.set_ylabel("Average Units Sold per SKU-Day")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.08, f"{h:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "08_promotion_lift.png")
    plt.close(fig)

    metrics_summary["promo_lift"] = {
        "non_promo_avg": round(float(promo_agg.loc[0]), 2),
        "promo_avg": round(float(promo_agg.loc[1]), 2),
        "promo_lift_pct": promo_lift_pct,
    }

    # -------------------------------------------------------------
    # 9. Stockout Analysis by Category
    # -------------------------------------------------------------
    sales_cat = pd.merge(sales_daily, sku_master[["sku_id", "category"]], on="sku_id", how="left")
    cat_stockout = (
        sales_cat.groupby("category")
        .agg(total_days=("stockout_flag", "count"), stockout_days=("stockout_flag", "sum"))
    )
    cat_stockout["stockout_rate_pct"] = (cat_stockout["stockout_days"] / cat_stockout["total_days"] * 100).round(2)
    cat_stockout = cat_stockout.sort_values("stockout_rate_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(cat_stockout.index, cat_stockout["stockout_rate_pct"], color=CRIMSON)
    ax.set_title("Stockout Incidence Rate by Product Category (%)")
    ax.set_xlabel("Percentage of SKU-Days in Stockout (%)")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.4, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "09_stockout_analysis.png")
    plt.close(fig)

    metrics_summary["stockout_rates_by_category"] = cat_stockout["stockout_rate_pct"].to_dict()

    # -------------------------------------------------------------
    # 10. Overstock & Days of Cover Analysis
    # -------------------------------------------------------------
    inv_cat = pd.merge(inventory_snapshots, sku_master[["sku_id", "category"]], on="sku_id", how="left")
    cat_overstock = (
        inv_cat.groupby("category")
        .apply(lambda g: pd.Series({
            "mean_doc": g["days_of_cover"].mean(),
            "overstock_rate_pct": (g["days_of_cover"] > 70).mean() * 100.0,
        }))
        .sort_values("overstock_rate_pct", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(cat_overstock.index, cat_overstock["overstock_rate_pct"], color=PURPLE)
    ax.set_title("Overstock Snapshot Exposure Rate (>70 Days Cover)")
    ax.set_xlabel("Percentage of Snapshots Overstocked (%)")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "10_overstock_analysis.png")
    plt.close(fig)

    metrics_summary["overstock_rates_by_category"] = cat_overstock["overstock_rate_pct"].round(2).to_dict()

    # -------------------------------------------------------------
    # 11. Inventory Value by Category
    # -------------------------------------------------------------
    # Use latest available snapshot date to show current warehouse capital tied up
    latest_snapshot_date = inventory_snapshots["snapshot_date"].max()
    latest_inv = inventory_snapshots[inventory_snapshots["snapshot_date"] == latest_snapshot_date]
    latest_inv_cat = pd.merge(latest_inv, sku_master[["sku_id", "category"]], on="sku_id", how="left")
    cat_inv_val = (
        latest_inv_cat.groupby("category")["inventory_value"].sum().sort_values(ascending=False) / 1e6
    )  # Millions

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(cat_inv_val.index, cat_inv_val.values, color=PRIMARY)
    ax.set_title(f"Warehouse Inventory Value at Horizon Close ({latest_snapshot_date})")
    ax.set_xlabel("Tied-up Working Capital (₹ Millions)")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2, f"₹{w:.2f}M", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "11_inventory_value_by_category.png")
    plt.close(fig)

    metrics_summary["latest_inventory_value"] = {
        "snapshot_date": str(latest_snapshot_date),
        "total_warehouse_value_inr": round(float(latest_inv["inventory_value"].sum()), 2),
        "category_shares_million_inr": cat_inv_val.round(2).to_dict(),
    }

    # Save summary metrics to JSON
    json_path = REPORTS_DIR / "eda_summary_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info("EDA analysis complete. 11 charts generated and metrics saved to %s", json_path)

    return metrics_summary


if __name__ == "__main__":
    sales = pd.read_csv(PROCESSED_DATA_DIR / "clean_sales_daily.csv")
    sku = pd.read_csv(PROCESSED_DATA_DIR / "clean_sku_master.csv")
    cal = pd.read_csv(PROCESSED_DATA_DIR / "clean_calendar.csv")
    inv = pd.read_csv(PROCESSED_DATA_DIR / "clean_inventory_snapshots.csv")
    weekly = pd.read_csv(PROCESSED_DATA_DIR / "weekly_demand.csv")
    generate_all_eda_charts(sales, sku, cal, inv, weekly)
