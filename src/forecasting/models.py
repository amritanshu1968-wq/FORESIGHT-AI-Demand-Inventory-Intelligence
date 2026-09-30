"""
PROJECT FORESIGHT — Machine Learning Forecasting Models
Implements Gradient Boosting & HistGradientBoosting models with scale denormalization and uncertainty intervals.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

from config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("forecasting_models")


class MLForecaster:
    """
    Wrapper for ML regression forecasters trained on scale-normalized features.
    Computes point forecasts and calibrated 80% empirical prediction intervals.
    """

    def __init__(
        self,
        model_type: str = "hist_gbm",
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.residual_std_ = 1.0  # Normalized residual standard error

        if model_type == "hist_gbm":
            self.model = HistGradientBoostingRegressor(
                max_iter=200,
                max_depth=5,
                learning_rate=0.06,
                min_samples_leaf=20,
                random_state=random_state,
            )
        elif model_type == "gbm":
            self.model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.07,
                subsample=0.85,
                random_state=random_state,
            )
        elif model_type == "ridge":
            self.model = Ridge(alpha=1.0, random_state=random_state)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLForecaster":
        """Fits model on normalized features and scaled target."""
        self.model.fit(X, y)
        preds = self.model.predict(X)
        residuals = y.values - preds
        self.residual_std_ = float(np.std(residuals))
        return self

    def predict(
        self,
        X: pd.DataFrame,
        sku_scale: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> np.ndarray:
        """Generates point forecasts rescaled back to natural unit scale."""
        scaled_preds = self.model.predict(X)
        if sku_scale is not None:
            natural_preds = np.asarray(scaled_preds) * np.asarray(sku_scale)
        else:
            natural_preds = scaled_preds
        return np.clip(natural_preds, 0.0, None)

    def predict_with_intervals(
        self,
        X: pd.DataFrame,
        sku_scale: Optional[Union[np.ndarray, pd.Series]] = None,
        confidence_level: float = 0.80,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes point forecasts and empirical prediction intervals:
        Interval = y_hat +/- Z * residual_std * scale
        80% confidence level corresponds to Z = 1.282
        """
        point_preds = self.predict(X, sku_scale=sku_scale)

        z_scores = {0.80: 1.282, 0.90: 1.645, 0.95: 1.96}
        z = z_scores.get(confidence_level, 1.282)

        if sku_scale is not None:
            margin = z * self.residual_std_ * np.asarray(sku_scale)
        else:
            margin = z * self.residual_std_

        lower_bounds = np.clip(point_preds - margin, 0.0, None)
        upper_bounds = np.clip(point_preds + margin, 0.0, None)

        return point_preds, lower_bounds, upper_bounds

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Serializes forecaster instance to disk."""
        if filepath is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = MODELS_DIR / f"{self.model_type}_forecaster.joblib"
        joblib.dump(self, filepath)
        logger.info("Forecaster saved to %s", filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "MLForecaster":
        """Loads serialized forecaster."""
        forecaster = joblib.load(filepath)
        return forecaster
