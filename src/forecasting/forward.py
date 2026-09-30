"""
PROJECT FORESIGHT — Forward Forecast Generator
Trains the selected production model on all historical data and produces recursive multi-step
forward forecasts with calibrated 80% empirical prediction intervals for every SKU.
"""

from datetime import timedelta
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

from config import FORECAST_HORIZON_WEEKS, FORECASTS_DIR, MODELS_DIR
from src.forecasting.features import engineer_sku_features, get_feature_columns, prepare_normalized_features
from src.forecasting.models import MLForecaster
from src.utils.logger import get_logger

logger = get_logger("forward_forecast")


def generate_forward_forecasts(
    weekly_df: pd.DataFrame,
    horizon_weeks: int = FORECAST_HORIZON_WEEKS,
    model_type: str = "hist_gbm",
    save_csv: bool = True,
) -> Tuple[pd.DataFrame, MLForecaster]:
    """
    Produces 12-week forward forecasts for all active SKUs.
    Saves outputs to outputs/forecasts/forecast_results.csv.
    """
    logger.info("Generating forward %d-week forecasts using %s...", horizon_weeks, model_type)

    feature_cols = get_feature_columns()
    feat_df = engineer_sku_features(weekly_df)

    # Train model on all fully lagged historical records
    latest_date_str = str(feat_df["week_start"].max())
    train_df, _, sku_scales = prepare_normalized_features(
        feature_df=feat_df,
        train_cutoff_date=latest_date_str,
        feature_cols=feature_cols,
    )

    model = MLForecaster(model_type=model_type, random_state=42)
    model.fit(train_df[feature_cols], train_df["scaled_target"])
    model.save(MODELS_DIR / "production_forecaster.joblib")

    # Generate forward weekly calendar dates
    last_week_dt = pd.to_datetime(latest_date_str)
    future_dates = [
        (last_week_dt + timedelta(weeks=w)).strftime("%Y-%m-%d")
        for w in range(1, horizon_weeks + 1)
    ]

    forecast_records: List[dict] = []

    # Map category codes
    category_map = {cat: i for i, cat in enumerate(weekly_df["category"].unique())}

    # Recursive forward prediction per SKU
    all_skus = weekly_df["sku_id"].unique()
    for sku in all_skus:
        sku_hist = weekly_df[weekly_df["sku_id"] == sku].copy()
        cat = sku_hist["category"].iloc[0]
        cat_code = category_map.get(cat, 0)
        scale = sku_scales.get(sku, 1.0)

        # Series of historical units and dates
        history_series = pd.Series(
            sku_hist["weekly_units_sold"].values,
            index=pd.to_datetime(sku_hist["week_start"]),
        )

        for week_idx, f_date_str in enumerate(future_dates, start=1):
            f_dt = pd.to_datetime(f_date_str)
            iso_cal = f_dt.isocalendar()
            wkn = int(iso_cal.week)
            mo = int(f_dt.month)
            quarter = int(f_dt.quarter)

            # Lags from history_series
            # lag_1 is previous week, lag_2 is 2 weeks ago, etc.
            def get_lag(k: int) -> float:
                target_dt = f_dt - timedelta(weeks=k)
                return float(history_series.get(target_dt, history_series.iloc[-1]))

            lag_1 = get_lag(1)
            lag_2 = get_lag(2)
            lag_3 = get_lag(3)
            lag_4 = get_lag(4)
            lag_8 = get_lag(8)
            lag_13 = get_lag(13)
            lag_26 = get_lag(26)
            lag_52 = get_lag(52)

            # Rolling stats over last 4 and 13 weeks
            recent_4 = [get_lag(k) for k in range(1, 5)]
            recent_13 = [get_lag(k) for k in range(1, 14)]
            roll_mean_4 = float(np.mean(recent_4))
            roll_std_4 = float(np.std(recent_4)) if len(recent_4) > 1 else 0.0
            roll_mean_13 = float(np.mean(recent_13))
            roll_std_13 = float(np.std(recent_13)) if len(recent_13) > 1 else 0.0

            # Cyclical signals
            sin_w = np.sin(2 * np.pi * wkn / 52.18)
            cos_w = np.cos(2 * np.pi * wkn / 52.18)
            sin_m = np.sin(2 * np.pi * mo / 12.0)
            cos_m = np.cos(2 * np.pi * mo / 12.0)
            fest = 1 if wkn in [10, 11, 12, 33, 41, 42, 43, 44, 45, 46, 47, 51, 52] else 0

            # Forward promotional assumptions: standard baseline (0) unless seasonal peak
            promo_days = 2 if fest == 1 else 0
            avg_disc = 0.15 if fest == 1 else 0.0
            is_promo = 1 if promo_days >= 2 else 0

            # Build single-row feature dataframe
            feat_row = pd.DataFrame([{
                "lag_1": lag_1 / scale,
                "lag_2": lag_2 / scale,
                "lag_3": lag_3 / scale,
                "lag_4": lag_4 / scale,
                "lag_8": lag_8 / scale,
                "lag_13": lag_13 / scale,
                "lag_26": lag_26 / scale,
                "lag_52": lag_52 / scale,
                "roll_mean_4": roll_mean_4 / scale,
                "roll_std_4": roll_std_4 / scale,
                "roll_mean_13": roll_mean_13 / scale,
                "roll_std_13": roll_std_13 / scale,
                "sin_week": sin_w,
                "cos_week": cos_w,
                "sin_month": sin_m,
                "cos_month": cos_m,
                "festival_window": fest,
                "promotion_days": promo_days,
                "average_discount": avg_disc,
                "is_promotional_week": is_promo,
                "category_code": cat_code,
            }])

            # Predict with intervals
            pred, lower, upper = model.predict_with_intervals(
                feat_row[feature_cols], sku_scale=scale, confidence_level=0.80
            )

            p_val = round(float(pred[0]), 2)
            l_val = round(float(lower[0]), 2)
            u_val = round(float(upper[0]), 2)

            forecast_records.append({
                "sku_id": sku,
                "category": cat,
                "forecast_week": f_date_str,
                "horizon_step": week_idx,
                "forecast_units": p_val,
                "lower_bound": l_val,
                "upper_bound": u_val,
                "model_name": model_type,
            })

            # Append prediction to history_series for subsequent recursive steps
            history_series[f_dt] = p_val

    forecast_df = pd.DataFrame(forecast_records)

    if save_csv:
        FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = FORECASTS_DIR / "forecast_results.csv"
        forecast_df.to_csv(out_csv, index=False)
        logger.info("Saved %d forward forecasts to %s", len(forecast_df), out_csv)

    return forecast_df, model
