import pytest
import numpy as np
import pandas as pd

from src.forecasting.metrics import calculate_metrics
from src.forecasting.features import engineer_sku_features, get_feature_columns, prepare_normalized_features
from src.forecasting.baseline import SeasonalNaiveForecaster, MovingAverageForecaster, LastWeekNaiveForecaster


def test_calculate_metrics_exact_values():
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    forecast = np.array([12.0, 18.0, 35.0, 35.0])
    # abs errors: 2, 2, 5, 5 -> sum = 14
    # sum actual = 100
    # WAPE = 14 / 100 * 100 = 14.0%
    # MAE = 14 / 4 = 3.5
    # sq errors: 4, 4, 25, 25 -> mean = 14.5 -> RMSE = sqrt(14.5) ~ 3.80788 ~ 3.81
    # sum forecast = 100 -> Bias = 0.0%
    met = calculate_metrics(actual, forecast)
    assert met["WAPE"] == 14.0
    assert met["MAE"] == 3.5
    assert met["RMSE"] == 3.81
    assert met["Bias"] == 0.0


def test_feature_engineering_leak_free():
    # Construct a synthetic SKU time-series
    dates = pd.date_range("2023-01-02", periods=60, freq="W-MON")
    units = np.arange(1, 61)  # 1, 2, 3, ... 60
    df = pd.DataFrame({
        "sku_id": "SKU-001",
        "week_start": dates.strftime("%Y-%m-%d"),
        "weekly_units_sold": units,
        "category": "Home Furnishings",
    })

    feat = engineer_sku_features(df)

    # 1. Check lag 1 at index 10: target at idx 10 is 11, lag_1 must be target at idx 9 (10)
    assert feat.loc[10, "lag_1"] == 10
    assert feat.loc[10, "lag_2"] == 9
    assert feat.loc[10, "lag_4"] == 7

    # 2. Check roll_mean_4 at index 10:
    # Must be mean of targets at idx 9, 8, 7, 6 -> (10 + 9 + 8 + 7) / 4 = 8.5
    assert feat.loc[10, "roll_mean_4"] == 8.5

    # 3. Target value at index 10 (11) must NOT appear in lag_1 or roll_mean_4 at index 10
    assert feat.loc[10, "lag_1"] != feat.loc[10, "weekly_units_sold"]
    assert feat.loc[10, "roll_mean_4"] != feat.loc[10, "weekly_units_sold"]


def test_baselines_correctness():
    df = pd.DataFrame({
        "lag_1": [10.0, 20.0],
        "roll_mean_4": [15.0, 25.0],
        "lag_52": [12.0, 22.0],
        "sku_scale": [1.0, 1.0],
    })

    s_naive = SeasonalNaiveForecaster()
    pred_s = s_naive.predict(df)
    assert np.allclose(pred_s, [12.0, 22.0])

    m_avg = MovingAverageForecaster()
    pred_m = m_avg.predict(df)
    assert np.allclose(pred_m, [15.0, 25.0])

    l_naive = LastWeekNaiveForecaster()
    pred_l = l_naive.predict(df)
    assert np.allclose(pred_l, [10.0, 20.0])


def test_scale_normalization_isolation():
    dates = pd.date_range("2023-01-02", periods=70, freq="W-MON")
    df = pd.DataFrame({
        "sku_id": "SKU-001",
        "week_start": dates.strftime("%Y-%m-%d"),
        "weekly_units_sold": [10.0] * 50 + [100.0] * 20,  # Huge surge in future!
        "category": "Home Furnishings",
    })
    feat = engineer_sku_features(df)
    cutoff = dates[49].strftime("%Y-%m-%d")

    train_df, test_df, scales = prepare_normalized_features(feat, train_cutoff_date=cutoff)

    # Scale must be exactly 10.0 (the mean of the first 50 weeks), NOT influenced by the 100.0 in the test set!
    assert scales["SKU-001"] == 10.0
    assert test_df["sku_scale"].iloc[0] == 10.0
