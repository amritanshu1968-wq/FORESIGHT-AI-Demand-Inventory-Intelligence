"""
PROJECT FORESIGHT — Real Data Ingestion Adapter
Provides an interface/template to load external or client-provided CSV/database extracts.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from config import RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("real_loader")


class RealDataLoader:
    """
    Adapter for ingesting client-provided raw extracts matching the FORESIGHT schema.
    Validates that incoming external files adhere to expected column structures.
    """

    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or RAW_DATA_DIR

    def load_sku_master(self, filename: str = "sku_master.csv") -> pd.DataFrame:
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Real data file not found: {path}")
        logger.info("Ingesting raw sku_master from %s", path)
        return pd.read_csv(path)

    def load_calendar(self, filename: str = "calendar.csv") -> pd.DataFrame:
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Real data file not found: {path}")
        logger.info("Ingesting raw calendar from %s", path)
        return pd.read_csv(path)

    def load_sales_daily(self, filename: str = "sales_daily.csv") -> pd.DataFrame:
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Real data file not found: {path}")
        logger.info("Ingesting raw sales_daily from %s", path)
        return pd.read_csv(path)

    def load_inventory_snapshots(self, filename: str = "inventory_snapshots.csv") -> pd.DataFrame:
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Real data file not found: {path}")
        logger.info("Ingesting raw inventory_snapshots from %s", path)
        return pd.read_csv(path)

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            self.load_sku_master(),
            self.load_calendar(),
            self.load_sales_daily(),
            self.load_inventory_snapshots(),
        )
