
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from config import DATASET_DISCLAIMER, RISK_DIR, SYNTHETIC_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("risk_service")


class RiskScoringService:
    def __init__(self, risk_csv_path: Optional[Path] = None):
        self.risk_path = risk_csv_path or (RISK_DIR / "risk_register.csv")
        self.risk_data: Optional[pd.DataFrame] = None
        self.sku_lookup: Dict[str, dict] = {}
        self.load_data()

    def load_data(self) -> bool:
        """Loads and indexes the validated risk register."""
        if not self.risk_path.exists():
            logger.warning("Risk register not found at %s", self.risk_path)
            self.risk_data = None
            self.sku_lookup = {}
            return False

        try:
            self.risk_data = pd.read_csv(self.risk_path)
            # Create fast O(1) dictionary index by sku_id
            self.sku_lookup = self.risk_data.set_index("sku_id").to_dict(orient="index")
            logger.info("RiskScoringService loaded %d active SKU risk records.", len(self.sku_lookup))
            return True
        except Exception as e:
            logger.error("Failed to load risk register: %s", e)
            self.risk_data = None
            self.sku_lookup = {}
            return False

    def is_loaded(self) -> bool:
        return self.risk_data is not None and len(self.sku_lookup) > 0

    def get_total_skus(self) -> int:
        return len(self.sku_lookup)

    def get_risk_score(self, sku_id: str) -> Optional[dict]:
        """Returns risk score record for a given SKU."""
        raw = self.sku_lookup.get(sku_id)
        if not raw:
            return None

        return {
            "sku_id": sku_id,
            "product_name": str(raw["product_name"]),
            "category": str(raw["category"]),
            "decision_quadrant": str(raw["risk_quadrant"]),
            "stockout_risk_score": float(raw["stockout_risk_score"]),
            "overstock_risk_score": float(raw["overstock_risk_score"]),
            "current_on_hand": int(raw["current_on_hand"]),
            "on_order_units": int(raw["on_order_units"]),
            "weeks_of_cover": float(raw["weeks_of_cover"]),
            "safety_stock": int(raw["safety_stock"]),
            "reorder_point": int(raw["reorder_point"]),
            "recommended_order_qty": int(raw["recommended_order_qty"]),
            "excess_units": int(raw["excess_units"]),
            "lost_margin_risk": float(raw["lost_margin_risk"]),
            "excess_capital_locked": float(raw["excess_capital_locked"]),
            "action_priority": str(raw["priority"]),
            "recommended_action": str(raw["recommended_action"]),
            "data_disclaimer": DATASET_DISCLAIMER,
        }

    def get_sku_decision(self, sku_id: str, mean_forecast: float = 0.0) -> Optional[dict]:
        """Returns combined product, inventory, risk, and action view for a SKU."""
        raw = self.sku_lookup.get(sku_id)
        if not raw:
            return None

        unit_p = float(raw["unit_price"])
        unit_c = float(raw["unit_cost"])
        gross_margin_pct = round((unit_p - unit_c) / unit_p * 100, 2) if unit_p > 0 else 0.0

        return {
            "sku_id": sku_id,
            "product": {
                "product_name": str(raw["product_name"]),
                "category": str(raw["category"]),
                "unit_price": unit_p,
                "unit_cost": unit_c,
                "gross_margin_pct": gross_margin_pct,
                "lead_time_days": int(raw["lead_time_days"]),
            },
            "inventory": {
                "current_on_hand": int(raw["current_on_hand"]),
                "on_order_units": int(raw["on_order_units"]),
                "weeks_of_cover": float(raw["weeks_of_cover"]),
                "safety_stock": int(raw["safety_stock"]),
                "reorder_point": int(raw["reorder_point"]),
                "inventory_value": float(raw["inventory_value"]),
            },
            "forecast_summary": {
                "forecast_weekly_demand": float(raw["forecast_weekly_demand"]),
                "projected_12w_demand": round(float(raw["forecast_weekly_demand"]) * 12, 1),
            },
            "risk": {
                "stockout_risk_score": float(raw["stockout_risk_score"]),
                "overstock_risk_score": float(raw["overstock_risk_score"]),
                "decision_quadrant": str(raw["risk_quadrant"]),
            },
            "financial_impact": {
                "lost_margin_risk": float(raw["lost_margin_risk"]),
                "excess_capital_locked": float(raw["excess_capital_locked"]),
                "total_financial_exposure": float(raw["total_financial_exposure"]),
            },
            "action": {
                "action_priority": str(raw["priority"]),
                "recommended_order_qty": int(raw["recommended_order_qty"]),
                "excess_units": int(raw["excess_units"]),
                "recommended_action": str(raw["recommended_action"]),
            },
            "data_disclaimer": DATASET_DISCLAIMER,
        }

    def batch_score(self, sku_ids: List[str]) -> Tuple[List[dict], List[str]]:
        """Scores multiple SKUs in batch."""
        results = []
        unmatched = []

        # Deduplicate while preserving order
        seen = set()
        unique_skus = []
        for s in sku_ids:
            if s not in seen:
                seen.add(s)
                unique_skus.append(s)

        for s_id in unique_skus:
            rec = self.sku_lookup.get(s_id)
            if rec:
                results.append({
                    "sku_id": s_id,
                    "product_name": str(rec["product_name"]),
                    "category": str(rec["category"]),
                    "decision_quadrant": str(rec["risk_quadrant"]),
                    "action_priority": str(rec["priority"]),
                    "stockout_risk_score": float(rec["stockout_risk_score"]),
                    "overstock_risk_score": float(rec["overstock_risk_score"]),
                    "recommended_order_qty": int(rec["recommended_order_qty"]),
                    "excess_units": int(rec["excess_units"]),
                    "lost_margin_risk": float(rec["lost_margin_risk"]),
                    "excess_capital_locked": float(rec["excess_capital_locked"]),
                    "recommended_action": str(rec["recommended_action"]),
                })
            else:
                unmatched.append(s_id)

        return results, unmatched
