"""
PROJECT FORESIGHT — Feature Engineering Module
Builds leakage-free lag, rolling, calendar, and promotional features for SKU-level demand.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("features")


def engineer_sku_features(
    weekly_df: pd.DataFrame,
    sku_col: str = "sku_id",
    date_col: str = "week_start",
    target_col: str = "weekly_units_sold",
) -> pd.DataFrame:
    """
    Constructs leak-free feature vectors for each SKU time-series.
    Every feature at time t uses ONLY information strictly prior to or known at time t.
    """
    df = weekly_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([sku_col, date_col]).reset_index(drop=True)

    # 1. Calendar & Cyclical Features (Known in advance at time t)
    dt_series = df[date_col]
    iso_cal = dt_series.dt.isocalendar()
    df["week_of_year"] = iso_cal.week.astype(int)
    df["year"] = iso_cal.year.astype(int)
    df["month"] = dt_series.dt.month.astype(int)
    df["quarter"] = dt_series.dt.quarter.astype(int)

    # Cyclical harmonics
    wkn = df["week_of_year"].values
    df["sin_week"] = np.sin(2 * np.pi * wkn / 52.18).round(4)
    df["cos_week"] = np.cos(2 * np.pi * wkn / 52.18).round(4)
    mo = df["month"].values
    df["sin_month"] = np.sin(2 * np.pi * mo / 12.0).round(4)
    df["cos_month"] = np.cos(2 * np.pi * mo / 12.0).round(4)

    # Festival period indicator (Weeks 41-47 Diwali peak, weeks 10-12 Holi, week 33 Independence Day)
    df["festival_window"] = np.isin(wkn, [10, 11, 12, 33, 41, 42, 43, 44, 45, 46, 47, 51, 52]).astype(int)

    # 2. Demand Lag Features (Strictly shifted past observations)
    # Lags: 1, 2, 3, 4, 8, 13, 26, 52
    lags = [1, 2, 3, 4, 8, 13, 26, 52]
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(sku_col)[target_col].shift(lag)

    # 3. Rolling Statistics (Strictly shifted by 1 to exclude current week target!)
    # roll_mean_4 = mean of y(t-1), y(t-2), y(t-3), y(t-4)
    shifted_target = df.groupby(sku_col)[target_col].shift(1)
    df["roll_mean_4"] = (
        shifted_target.groupby(df[sku_col]).rolling(window=4, min_periods=1).mean().values
    )
    df["roll_std_4"] = (
        shifted_target.groupby(df[sku_col]).rolling(window=4, min_periods=1).std().fillna(0).values
    )
    df["roll_mean_13"] = (
        shifted_target.groupby(df[sku_col]).rolling(window=13, min_periods=2).mean().values
    )
    df["roll_std_13"] = (
        shifted_target.groupby(df[sku_col]).rolling(window=13, min_periods=2).std().fillna(0).values
    )

    # 4. Business & Promotional Features (Known for current week planning)
    if "promotion_days" not in df.columns:
        df["promotion_days"] = 0
    if "average_discount" not in df.columns:
        df["average_discount"] = 0.0
    if "is_promotional_week" not in df.columns:
        df["is_promotional_week"] = (df["promotion_days"] >= 2).astype(int)

    return df


def get_feature_columns() -> List[str]:
    """Returns the list of input feature names used for model training."""
    return [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "lag_8",
        "lag_13",
        "lag_26",
        "lag_52",
        "roll_mean_4",
        "roll_std_4",
        "roll_mean_13",
        "roll_std_13",
        "sin_week",
        "cos_week",
        "sin_month",
        "cos_month",
        "festival_window",
        "promotion_days",
        "average_discount",
        "is_promotional_week",
        "category_code",
    ]


def prepare_normalized_features(
    feature_df: pd.DataFrame,
    train_cutoff_date: str,
    feature_cols: List[str] = None,
    target_col: str = "weekly_units_sold",
    sku_col: str = "sku_id",
    date_col: str = "week_start",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Computes per-SKU historical scale strictly on data <= train_cutoff_date (zero leakage),
    normalizes lag/target columns by scale, and splits into train and test sets.
    """
    df = feature_df.copy()

    # Category code encoding
    if "category" in df.columns:
        df["category_code"] = df["category"].astype("category").cat.codes
    elif "category_code" not in df.columns:
        df["category_code"] = 0

    if feature_cols is None:
        feature_cols = get_feature_columns()

    # 1. Compute per-SKU scale strictly using training slice
    train_mask = df[date_col] <= train_cutoff_date
    sku_scales = (
        df.loc[train_mask]
        .groupby(sku_col)[target_col]
        .mean()
        .clip(lower=0.5)  # Avoid division by zero for zero-demand SKUs
    )

    # Map scale to entire dataframe
    df["sku_scale"] = df[sku_col].map(sku_scales).fillna(1.0)

    # 2. Scale demand-related features and target
    scale_cols = [c for c in feature_cols if c.startswith("lag_") or c.startswith("roll_")]
    for c in scale_cols:
        df[c] = (df[c] / df["sku_scale"]).fillna(0.0)

    df["scaled_target"] = df[target_col] / df["sku_scale"]

    # Split train and test
    train_df = df[train_mask & df["lag_52"].notna()].copy()
    test_df = df[~train_mask].copy()

    return train_df, test_df, sku_scales
