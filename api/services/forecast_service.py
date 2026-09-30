

from pathlib import Path
from typing import Dict, List, Optional
import joblib
import pandas as pd

from config import DATASET_DISCLAIMER, FORECASTS_DIR, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("forecast_service")


class ForecastService:
    def __init__(
        self,
        forecast_csv_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
    ):
        self.forecast_path = forecast_csv_path or (FORECASTS_DIR / "forecast_results.csv")
        self.model_path = model_path or (MODELS_DIR / "production_forecaster.joblib")

        self.forecast_data: Optional[pd.DataFrame] = None
        self.sku_forecast_map: Dict[str, List[dict]] = {}
        self.model_asset = None

        self.load_data()
        self.load_model()

    def load_data(self) -> bool:
        """Loads and indexes forward forecast CSV artifact."""
        if not self.forecast_path.exists():
            logger.warning("Forecast artifact not found at %s", self.forecast_path)
            self.forecast_data = None
            self.sku_forecast_map = {}
            return False

        try:
            self.forecast_data = pd.read_csv(self.forecast_path)
            # Group by sku_id
            grouped = self.forecast_data.groupby("sku_id")
            self.sku_forecast_map = {}
            for sku_id, group in grouped:
                self.sku_forecast_map[sku_id] = group.sort_values("horizon_step").to_dict(orient="records")
            logger.info("ForecastService loaded forecasts for %d SKUs.", len(self.sku_forecast_map))
            return True
        except Exception as e:
            logger.error("Failed to load forecast results: %s", e)
            self.forecast_data = None
            self.sku_forecast_map = {}
            return False

    def load_model(self) -> bool:
        """Loads serialized production model asset without fitting."""
        if not self.model_path.exists():
            logger.warning("Production forecaster model not found at %s", self.model_path)
            self.model_asset = None
            return False

        try:
            self.model_asset = joblib.load(self.model_path)
            logger.info("ForecastService loaded serialized production forecaster joblib.")
            return True
        except Exception as e:
            logger.error("Failed to load model asset: %s", e)
            self.model_asset = None
            return False

    def is_loaded(self) -> bool:
        return self.forecast_data is not None and len(self.sku_forecast_map) > 0

    def is_model_loaded(self) -> bool:
        return self.model_asset is not None

    def get_forecast(self, sku_id: str) -> Optional[dict]:
        """Returns 12-week forward forecast series for a SKU."""
        records = self.sku_forecast_map.get(sku_id)
        if not records:
            return None

        first = records[0]
        points = [
            {
                "week": str(r["forecast_week"]),
                "horizon_step": int(r["horizon_step"]),
                "forecast_units": float(r["forecast_units"]),
                "lower_bound": float(r["lower_bound"]),
                "upper_bound": float(r["upper_bound"]),
            }
            for r in records
        ]

        return {
            "sku_id": sku_id,
            "category": str(first["category"]),
            "model_name": str(first["model_name"]),
            "forecast_horizon_weeks": len(points),
            "forecast": points,
            "data_disclaimer": DATASET_DISCLAIMER,
        }
