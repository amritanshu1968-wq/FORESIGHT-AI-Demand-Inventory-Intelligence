"""
PROJECT FORESIGHT — Baseline Forecasting Models
Implements Seasonal Naive (52-week lag), 4-Week Moving Average, and Last-Week Naive baselines.
"""

from typing import Union
import numpy as np
import pandas as pd


class SeasonalNaiveForecaster:
    """
    Seasonal Naive Baseline:
    Forecasts demand using the same calendar week from 52 weeks prior: y_hat(t) = y(t - 52).
    Fallback strategy: If 52-week lag is missing, fall back to y(t - 1) (last week).
    """

    def __init__(self, fallback_col: str = "lag_1"):
        self.fallback_col = fallback_col

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # Check if unscaled lag_52 is present, else use scaled lag_52 * sku_scale
        if "lag_52" in df.columns:
            if "sku_scale" in df.columns:
                preds = (df["lag_52"] * df["sku_scale"]).values.copy()
                fallback = (df[self.fallback_col] * df["sku_scale"]).values.copy()
            else:
                preds = df["lag_52"].values.copy()
                fallback = df[self.fallback_col].values.copy()

            # Where lag_52 is NaN or negative, use fallback
            nan_mask = np.isnan(preds) | (preds < 0)
            preds[nan_mask] = fallback[nan_mask]
            return np.nan_to_num(preds, nan=0.0)
        raise ValueError("Dataframe must contain 'lag_52' column for Seasonal Naive forecast.")


class MovingAverageForecaster:
    """
    Moving Average Baseline:
    Forecasts demand as the mean of the preceding 4 weeks: y_hat(t) = mean(y(t-1) ... y(t-4)).
    """

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if "roll_mean_4" in df.columns:
            if "sku_scale" in df.columns:
                preds = (df["roll_mean_4"] * df["sku_scale"]).values
            else:
                preds = df["roll_mean_4"].values
            return np.nan_to_num(preds, nan=0.0)
        raise ValueError("Dataframe must contain 'roll_mean_4' column for Moving Average forecast.")


class LastWeekNaiveForecaster:
    """
    Last-Week Naive Baseline:
    Forecasts demand equal to the immediately preceding week: y_hat(t) = y(t - 1).
    """

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if "lag_1" in df.columns:
            if "sku_scale" in df.columns:
                preds = (df["lag_1"] * df["sku_scale"]).values
            else:
                preds = df["lag_1"].values
            return np.nan_to_num(preds, nan=0.0)
        raise ValueError("Dataframe must contain 'lag_1' column for Last-Week Naive forecast.")
