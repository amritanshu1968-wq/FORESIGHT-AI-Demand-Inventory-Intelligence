"""
PROJECT FORESIGHT — Forecasting Evaluation Charts
Generates publication-quality charts for model comparison, backtesting folds, and holdout fits.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CHARTS_DIR
from src.utils.logger import get_logger

logger = get_logger("evaluation_plots")

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


def generate_forecasting_charts(
    backtest_results: pd.DataFrame,
    comparison_df: pd.DataFrame,
    weekly_demand: pd.DataFrame,
) -> None:
    """Generates all 4 forecast evaluation charts."""
    logger.info("Generating forecasting evaluation charts...")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Model Comparison Chart (WAPE Bar Chart)
    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_comp = comparison_df.sort_values("WAPE", ascending=False)
    colors = [EMERALD if m == comparison_df.iloc[0]["Model"] else PRIMARY for m in sorted_comp["Model"]]
    bars = ax.barh(sorted_comp["Model"], sorted_comp["WAPE"], color=colors, height=0.55)
    ax.set_title("Walk-Forward Model Comparison (WAPE % — Lower is Better)")
    ax.set_xlabel("WAPE (%)")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(sorted_comp["WAPE"]) * 1.18)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "model_comparison.png")
    plt.close(fig)

    # 2. Fold WAPE Stability Chart
    fold_perf = []
    models = ["Seasonal Naive (52w)", "HistGradientBoosting", "Moving Average (4w)"]
    for fold in sorted(backtest_results["fold"].unique()):
        sub = backtest_results[backtest_results["fold"] == fold]
        y_act = sub["weekly_units_sold"].values
        sum_act = np.sum(y_act)
        for m, col in [
            ("Seasonal Naive (52w)", "pred_seasonal_naive"),
            ("HistGradientBoosting", "pred_hist_gbm"),
            ("Moving Average (4w)", "pred_moving_avg_4w"),
        ]:
            err = np.sum(np.abs(y_act - sub[col].values))
            wape = (err / sum_act * 100.0) if sum_act > 0 else 0.0
            fold_perf.append({"Fold": f"Fold {fold}", "Model": m, "WAPE": round(wape, 2)})

    fold_df = pd.DataFrame(fold_perf)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    pivot_fold = fold_df.pivot(index="Fold", columns="Model", values="WAPE")
    pivot_fold.plot(kind="bar", ax=ax, color=[EMERALD, ACCENT, AMBER], width=0.7)
    ax.set_title("WAPE Stability Across 4 Sequential Backtest Origins")
    ax.set_ylabel("WAPE (%)")
    ax.set_xlabel("Backtest Origin Window")
    ax.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#E2E8F0")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "fold_wape.png")
    plt.close(fig)

    # 3. Forecast vs Actual on Holdout for Top-Selling SKU
    top_sku = (
        weekly_demand.groupby("sku_id")["weekly_units_sold"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )
    # Take latest fold predictions for this SKU
    latest_fold = backtest_results["fold"].max()
    sku_fold = backtest_results[
        (backtest_results["sku_id"] == top_sku) & (backtest_results["fold"] == latest_fold)
    ].sort_values("week_start")

    # History prior to holdout
    sku_hist = weekly_demand[weekly_demand["sku_id"] == top_sku].copy()
    sku_hist["week_start"] = pd.to_datetime(sku_hist["week_start"])
    sku_fold = sku_fold.copy()
    sku_fold["week_start"] = pd.to_datetime(sku_fold["week_start"])
    holdout_start = sku_fold["week_start"].min()
    prior_hist = sku_hist[sku_hist["week_start"] < holdout_start].tail(24)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    # Historical Actuals
    ax.plot(
        pd.to_datetime(prior_hist["week_start"]),
        prior_hist["weekly_units_sold"],
        color=PRIMARY,
        lw=1.8,
        label="Historical Actuals",
    )
    # Holdout Actuals
    ax.plot(
        pd.to_datetime(sku_fold["week_start"]),
        sku_fold["weekly_units_sold"],
        color=PRIMARY,
        lw=2.2,
        ls="-",
        label="Test Actuals",
    )
    # HistGradientBoosting Forecast
    ax.plot(
        pd.to_datetime(sku_fold["week_start"]),
        sku_fold["pred_hist_gbm"],
        color=EMERALD,
        lw=2.2,
        marker="o",
        label="ML Forecast (HistGBM)",
    )
    # 80% Prediction Interval
    ax.fill_between(
        pd.to_datetime(sku_fold["week_start"]),
        sku_fold["lower_bound_hist_gbm"],
        sku_fold["upper_bound_hist_gbm"],
        color=EMERALD,
        alpha=0.18,
        label="80% Prediction Interval",
    )
    # Seasonal Naive
    ax.plot(
        pd.to_datetime(sku_fold["week_start"]),
        sku_fold["pred_seasonal_naive"],
        color=AMBER,
        lw=1.5,
        ls="--",
        label="Seasonal Naive (52w)",
    )

    ax.axvline(pd.to_datetime(holdout_start), color=CRIMSON, ls=":", lw=1.5, label="Forecast Origin")
    ax.set_title(f"Walk-Forward Holdout: Actual vs. Forecast for Top Best-Seller ({top_sku})")
    ax.set_xlabel("Timeline")
    ax.set_ylabel("Weekly Demand (Units Sold)")
    ax.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#E2E8F0", loc="upper left")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "forecast_vs_actual.png")
    plt.close(fig)

    # 4. Category-Level Forecast Error
    cat_perf = []
    for cat in sorted(backtest_results["category"].unique()):
        sub_cat = backtest_results[backtest_results["category"] == cat]
        y_act = sub_cat["weekly_units_sold"].values
        sum_act = np.sum(y_act)
        for m, col in [
            ("Seasonal Naive", "pred_seasonal_naive"),
            ("HistGBM (ML)", "pred_hist_gbm"),
        ]:
            err = np.sum(np.abs(y_act - sub_cat[col].values))
            wape = (err / sum_act * 100.0) if sum_act > 0 else 0.0
            cat_perf.append({"Category": cat, "Model": m, "WAPE": round(wape, 2)})

    cat_df = pd.DataFrame(cat_perf)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    pivot_cat = cat_df.pivot(index="Category", columns="Model", values="WAPE")
    pivot_cat.plot(kind="bar", ax=ax, color=[AMBER, EMERALD], width=0.6)
    ax.set_title("Forecast Accuracy (WAPE %) by Product Category: ML vs. Baseline")
    ax.set_ylabel("WAPE (%) — Lower is Better")
    ax.set_xlabel("Product Category")
    ax.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#E2E8F0")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "category_forecast_error.png")
    plt.close(fig)

    logger.info("All 4 forecasting evaluation charts saved to %s", CHARTS_DIR)
