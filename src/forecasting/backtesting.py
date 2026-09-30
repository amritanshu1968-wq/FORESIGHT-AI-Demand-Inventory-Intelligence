"""
PROJECT FORESIGHT — Rolling-Origin Backtesting Engine
Executes strict walk-forward cross-validation without future data leakage.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from config import BACKTEST_SPLITS, BACKTEST_TEST_WEEKS, REPORTS_DIR
from src.forecasting.baseline import (
    LastWeekNaiveForecaster,
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
)
from src.forecasting.features import (
    engineer_sku_features,
    get_feature_columns,
    prepare_normalized_features,
)
from src.forecasting.metrics import calculate_metrics
from src.forecasting.models import MLForecaster
from src.utils.logger import get_logger

logger = get_logger("backtesting")


def run_rolling_origin_backtest(
    weekly_df: pd.DataFrame,
    n_splits: int = BACKTEST_SPLITS,
    horizon_weeks: int = BACKTEST_TEST_WEEKS,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes walk-forward backtesting across sequential origins.
    Evaluates:
    - Seasonal Naive (52w lag)
    - 4-Week Moving Average
    - Last-Week Naive
    - HistGradientBoosting (ML)
    - Ridge Regression
    """
    logger.info(
        "Starting %d-fold rolling-origin backtest with %d-week test horizons...",
        n_splits,
        horizon_weeks,
    )

    # 1. Feature Engineering across all historical periods
    feat_df = engineer_sku_features(weekly_df)
    feature_cols = get_feature_columns()

    # Get chronological distinct weekly dates
    unique_weeks = sorted(feat_df["week_start"].unique())
    total_weeks = len(unique_weeks)

    # Determine origins: Walk forward by horizon_weeks
    # e.g., if total_weeks=158, horizon=8, n_splits=4:
    # Fold 4: origin at week 150 (tests 151..158)
    # Fold 3: origin at week 142 (tests 143..150)
    # Fold 2: origin at week 134 (tests 135..142)
    # Fold 1: origin at week 126 (tests 127..134)
    origin_indices = [
        total_weeks - (n_splits - i) * horizon_weeks for i in range(n_splits)
    ]

    all_fold_predictions: List[pd.DataFrame] = []
    fold_metric_records: List[Dict[str, Any]] = []

    # Initialize models
    seasonal_naive = SeasonalNaiveForecaster()
    moving_avg = MovingAverageForecaster()
    last_week_naive = LastWeekNaiveForecaster()

    for fold_num, origin_idx in enumerate(origin_indices, start=1):
        origin_date = unique_weeks[origin_idx - 1]
        test_end_date = unique_weeks[min(origin_idx + horizon_weeks - 1, total_weeks - 1)]

        logger.info(
            "Fold %d/%d: Origin=%s | Test Window: %s to %s",
            fold_num,
            n_splits,
            origin_date,
            unique_weeks[origin_idx],
            test_end_date,
        )

        # Prepare normalized features with ZERO leakage (scale strictly <= origin_date)
        train_df, test_slice, sku_scales = prepare_normalized_features(
            feature_df=feat_df,
            train_cutoff_date=origin_date,
            feature_cols=feature_cols,
        )

        # Filter test_slice to only the current fold's forward horizon window
        fold_test = test_slice[
            (test_slice["week_start"] > origin_date)
            & (test_slice["week_start"] <= test_end_date)
        ].copy()

        if len(fold_test) == 0:
            logger.warning("Fold %d has empty test set, skipping.", fold_num)
            continue

        # Target actuals
        actuals = fold_test["weekly_units_sold"].values

        # 1. Baseline Predictions
        pred_s_naive = seasonal_naive.predict(fold_test)
        pred_m_avg = moving_avg.predict(fold_test)
        pred_last_w = last_week_naive.predict(fold_test)

        # 2. Train ML Models on fold's training slice
        # A. HistGradientBoosting
        ml_model = MLForecaster(model_type="hist_gbm", random_state=42)
        ml_model.fit(train_df[feature_cols], train_df["scaled_target"])
        pred_ml, lower_ml, upper_ml = ml_model.predict_with_intervals(
            fold_test[feature_cols], sku_scale=fold_test["sku_scale"]
        )

        # B. Ridge Regression
        ridge_model = MLForecaster(model_type="ridge", random_state=42)
        ridge_model.fit(train_df[feature_cols], train_df["scaled_target"])
        pred_ridge = ridge_model.predict(
            fold_test[feature_cols], sku_scale=fold_test["sku_scale"]
        )

        # Record predictions per row
        fold_res = fold_test[
            ["sku_id", "week_start", "weekly_units_sold", "category", "sku_scale"]
        ].copy()
        fold_res["fold"] = fold_num
        fold_res["origin_date"] = origin_date
        fold_res["pred_seasonal_naive"] = np.round(pred_s_naive, 2)
        fold_res["pred_moving_avg_4w"] = np.round(pred_m_avg, 2)
        fold_res["pred_last_week"] = np.round(pred_last_w, 2)
        fold_res["pred_hist_gbm"] = np.round(pred_ml, 2)
        fold_res["lower_bound_hist_gbm"] = np.round(lower_ml, 2)
        fold_res["upper_bound_hist_gbm"] = np.round(upper_ml, 2)
        fold_res["pred_ridge"] = np.round(pred_ridge, 2)

        all_fold_predictions.append(fold_res)

        # Evaluate fold metrics
        models_dict = {
            "Seasonal Naive (52w)": pred_s_naive,
            "Moving Average (4w)": pred_m_avg,
            "Last-Week Naive": pred_last_w,
            "HistGradientBoosting": pred_ml,
            "Ridge Regression": pred_ridge,
        }

        for m_name, preds in models_dict.items():
            metrics = calculate_metrics(actuals, preds)
            fold_metric_records.append({
                "fold": fold_num,
                "origin_date": origin_date,
                "model": m_name,
                **metrics,
            })

    # Combine all backtest predictions
    backtest_results = pd.concat(all_fold_predictions, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metric_records)

    # Compute overall pooled metrics across all folds
    overall_comparison = []
    models_to_compare = [
        ("Seasonal Naive (52w)", "pred_seasonal_naive"),
        ("Moving Average (4w)", "pred_moving_avg_4w"),
        ("Last-Week Naive", "pred_last_week"),
        ("HistGradientBoosting", "pred_hist_gbm"),
        ("Ridge Regression", "pred_ridge"),
    ]

    total_actuals = backtest_results["weekly_units_sold"].values
    for m_label, col in models_to_compare:
        m_preds = backtest_results[col].values
        met = calculate_metrics(total_actuals, m_preds)
        overall_comparison.append({"Model": m_label, **met})

    comparison_df = pd.DataFrame(overall_comparison).sort_values("WAPE").reset_index(drop=True)

    # Improvement vs Seasonal Naive
    s_naive_wape = comparison_df.loc[
        comparison_df["Model"] == "Seasonal Naive (52w)", "WAPE"
    ].values[0]
    best_model = comparison_df.iloc[0]["Model"]
    best_wape = comparison_df.iloc[0]["WAPE"]
    improvement_pct = round((1.0 - best_wape / s_naive_wape) * 100.0, 2)

    summary: Dict[str, Any] = {
        "n_splits": n_splits,
        "horizon_weeks": horizon_weeks,
        "total_evaluated_records": len(backtest_results),
        "best_model": best_model,
        "best_wape": best_wape,
        "seasonal_naive_wape": s_naive_wape,
        "wape_improvement_pct": improvement_pct,
        "comparison_table": comparison_df.to_dict(orient="records"),
    }

    # Save to outputs/
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    backtest_results.to_csv(REPORTS_DIR / "backtest_results.csv", index=False)
    comparison_df.to_csv(REPORTS_DIR / "forecast_model_comparison.csv", index=False)
    logger.info("Backtesting complete! Model Comparison:\n%s", comparison_df.to_string(index=False))

    return backtest_results, comparison_df, summary
