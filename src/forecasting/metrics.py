"""
PROJECT FORESIGHT — Forecasting Evaluation Metrics
Computes WAPE, MAE, RMSE, MAPE, and Signed Bias.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd


def calculate_metrics(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """
    Computes standard demand forecasting accuracy metrics.

    Primary:
    - WAPE (%): Weighted Absolute Percentage Error = sum(|y - y_hat|) / sum(y) * 100

    Secondary:
    - MAE: Mean Absolute Error = mean(|y - y_hat|)
    - RMSE: Root Mean Squared Error = sqrt(mean((y - y_hat)^2))
    - Signed Bias (%): sum(y_hat - y) / sum(y) * 100 (+ indicates over-forecasting, - under)
    - MAPE (%): Mean Absolute Percentage Error (computed on non-zero actuals)
    """
    y = np.asarray(actual, dtype=float)
    y_hat = np.asarray(forecast, dtype=float)

    if len(y) == 0:
        return {"WAPE": 0.0, "MAE": 0.0, "RMSE": 0.0, "Bias": 0.0, "MAPE": 0.0}

    # Absolute and squared errors
    abs_errors = np.abs(y - y_hat)
    sq_errors = (y - y_hat) ** 2
    sum_actual = np.sum(y)

    # 1. WAPE
    if sum_actual > 0:
        wape = float(np.sum(abs_errors) / sum_actual * 100.0)
    else:
        wape = 0.0 if np.sum(abs_errors) == 0 else 100.0

    # 2. MAE & RMSE
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))

    # 3. Signed Bias
    if sum_actual > 0:
        bias = float((np.sum(y_hat) - sum_actual) / sum_actual * 100.0)
    else:
        bias = 0.0

    # 4. MAPE (calculated on y > 0 to prevent division by zero explosion)
    nonzero_mask = y > 0
    if np.any(nonzero_mask):
        mape = float(np.mean(abs_errors[nonzero_mask] / y[nonzero_mask]) * 100.0)
    else:
        mape = 0.0

    return {
        "WAPE": round(wape, 2),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "Bias": round(bias, 2),
        "MAPE": round(mape, 2),
    }
