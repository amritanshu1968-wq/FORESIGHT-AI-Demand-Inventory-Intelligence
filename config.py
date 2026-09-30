
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
FORECASTS_DIR = OUTPUTS_DIR / "forecasts"
RISK_DIR = OUTPUTS_DIR / "risk"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
for directory in [
    RAW_DATA_DIR,
    SYNTHETIC_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHARTS_DIR,
    FORECASTS_DIR,
    RISK_DIR,
    REPORTS_DIR,
    MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Data Mode: "synthetic" (primary) or "real" (adapter)
DATA_MODE = "synthetic"

# Synthetic Generation Parameters
RANDOM_SEED = 42
N_SKUS = 300
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"  # 3 full calendar years (1096 days)

# Business & Inventory Policy Parameters
DEFAULT_SERVICE_LEVEL = 0.95  # Target service level for safety stock (Z ~ 1.645)
Z_SCORE_95 = 1.6449
OVERSTOCK_COVER_WEEKS_THRESHOLD = 10.0  # > 10 weeks of forward cover is considered overstocked
UNDERSTOCK_COVER_WEEKS_THRESHOLD = 2.0  # < 2 weeks of cover is imminent stockout risk
ANNUAL_HOLDING_COST_RATE = 0.20  # 20% annual holding cost rate on unit cost

# Forecasting Parameters
FORECAST_HORIZON_WEEKS = 12  # Standard forward planning horizon
BACKTEST_SPLITS = 4  # Number of rolling-origin walk-forward splits
BACKTEST_TEST_WEEKS = 8  # Holdout duration per rolling fold

# Categories & Subcategories for NorthBay Living (Home & Lifestyle)
PRODUCT_TAXONOMY = {
    "Home Furnishings": {
        "subcategories": ["Bedding", "Cushions & Throws", "Rugs & Mats", "Curtains"],
        "base_price_range": (800.0, 6500.0),
        "lead_time_days_range": (14, 30),
        "base_demand_scale": 1.2,
    },
    "Décor & Lighting": {
        "subcategories": ["Table Lamps", "Pendant Lights", "Wall Art", "Vases & Planters"],
        "base_price_range": (600.0, 8500.0),
        "lead_time_days_range": (10, 25),
        "base_demand_scale": 0.9,
    },
    "Kitchen & Dining": {
        "subcategories": ["Cast Iron Cookware", "Dinnerware Sets", "Glassware", "Cutlery Sets"],
        "base_price_range": (450.0, 5500.0),
        "lead_time_days_range": (12, 28),
        "base_demand_scale": 1.4,
    },
    "Small Appliances": {
        "subcategories": ["Espresso Machines", "Air Fryers", "Cold Press Juicers", "Electric Kettles"],
        "base_price_range": (1800.0, 16500.0),
        "lead_time_days_range": (20, 45),
        "base_demand_scale": 0.7,
    },
}

DATASET_DISCLAIMER = (
    "Data Source: Synthetic retail dataset generated for project development and demonstration. "
    "These findings demonstrate the capabilities of the FORESIGHT platform and should not be "
    "interpreted as actual NorthBay Living or Zidio Development business results."
)
