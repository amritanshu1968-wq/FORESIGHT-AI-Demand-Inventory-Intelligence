"""
PROJECT FORESIGHT — Phase 5 Orchestration & Inventory Risk Engine
Computes 2D risk scores, decision quadrants, EOQ/reorder recommendations,
rupee business impact, and compiles outputs/risk/risk_register.csv and reports.
"""

import json
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_DISCLAIMER, FORECASTS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, RISK_DIR, SYNTHETIC_DATA_DIR
from src.inventory.impact import assign_action_priority, calculate_rupee_impact
from src.inventory.recommendations import calculate_eoq, generate_sku_recommendation
from src.inventory.risk_engine import (
    assign_risk_quadrant,
    calculate_overstock_risk_score,
    calculate_reorder_point,
    calculate_safety_stock,
    calculate_stockout_risk_score,
    calculate_weeks_of_cover,
)
from src.inventory.risk_plots import plot_risk_decision_grid
from src.utils.logger import get_logger

logger = get_logger("run_phase5")


def build_inventory_risk_register(
    sku_master: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
    forecast_results: pd.DataFrame,
    backtest_results: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Computes SKU-level risk metrics, recommendations, and rupee impact for all active SKUs.
    """
    logger.info("Executing 2D inventory risk scoring across all active SKUs...")

    # 1. Latest inventory snapshot (2025-12-31)
    latest_date = inventory_snapshots["snapshot_date"].max()
    latest_inv = inventory_snapshots[inventory_snapshots["snapshot_date"] == latest_date].copy()
    inv_pos_map = latest_inv.set_index("sku_id")[["closing_stock", "on_order_units"]].to_dict(orient="index")

    # 2. Average forecast weekly demand over 12 forward weeks
    mean_forecast = (
        forecast_results.groupby("sku_id")["forecast_units"].mean().round(2).to_dict()
    )

    # 3. Residual error standard deviation from backtest results
    backtest_results["resid"] = backtest_results["weekly_units_sold"] - backtest_results["pred_hist_gbm"]
    sku_sigma_map = backtest_results.groupby("sku_id")["resid"].std().fillna(1.0).to_dict()

    risk_records = []

    for _, sku_row in sku_master.iterrows():
        sku_id = sku_row["sku_id"]
        product_name = sku_row["product_name"]
        category = sku_row["category"]
        unit_price = float(sku_row["unit_price"])
        unit_cost = float(sku_row["unit_cost"])
        lead_time_days = float(sku_row["lead_time_days"])
        lead_time_weeks = lead_time_days / 7.0
        service_level = float(sku_row.get("target_service_level", 0.95))
        moq = int(sku_row.get("minimum_order_qty", 15))

        # Current stock position
        pos = inv_pos_map.get(sku_id, {"closing_stock": 0, "on_order_units": 0})
        on_hand = int(pos["closing_stock"])
        on_order = int(pos["on_order_units"])
        effective_avail = on_hand + on_order

        # Forecast demand & error
        f_demand = max(float(mean_forecast.get(sku_id, 1.0)), 0.0)
        ltd = round(f_demand * lead_time_weeks, 2)
        sigma_err = float(sku_sigma_map.get(sku_id, max(f_demand * 0.25, 1.0)))

        # Safety Stock & ROP
        ss = calculate_safety_stock(lead_time_days, sigma_err, service_level)
        rop = calculate_reorder_point(ltd, ss)

        # Weeks of Cover
        woc = calculate_weeks_of_cover(on_hand, f_demand)

        # Demand Volatility (Coefficient of Variation)
        demand_cv = round(sigma_err / max(f_demand, 1.0), 3)

        # Risk Scores
        stockout_score = calculate_stockout_risk_score(effective_avail, ltd, rop, ss, demand_cv=demand_cv)
        overstock_score = calculate_overstock_risk_score(woc, lead_time_weeks, demand_cv=demand_cv)

        # Quadrant Assignment
        quadrant = assign_risk_quadrant(stockout_score, overstock_score)

        # EOQ
        annual_demand = f_demand * 52.0
        eoq = calculate_eoq(annual_demand, unit_cost, order_cost=500.0, moq=moq)

        # Recommendations
        action, rec_order_qty, excess_u = generate_sku_recommendation(
            quadrant=quadrant,
            current_on_hand=on_hand,
            effective_available=effective_avail,
            reorder_point=rop,
            forecast_weekly_demand=f_demand,
            lead_time_weeks=lead_time_weeks,
            eoq=eoq,
            moq=moq,
        )

        # Target stock level for overstock capital calculation
        target_stock = int(np.ceil(max(rop, (lead_time_weeks + 4.0) * f_demand)))

        # Rupee Impact
        lost_margin_risk, excess_capital_locked, inv_val, total_exposure = calculate_rupee_impact(
            effective_available=effective_avail,
            current_on_hand=on_hand,
            lead_time_demand=ltd,
            target_stock=target_stock,
            unit_price=unit_price,
            unit_cost=unit_cost,
        )

        risk_records.append({
            "sku_id": sku_id,
            "product_name": product_name,
            "category": category,
            "current_on_hand": on_hand,
            "on_order_units": on_order,
            "forecast_weekly_demand": f_demand,
            "lead_time_days": int(lead_time_days),
            "safety_stock": ss,
            "reorder_point": rop,
            "weeks_of_cover": woc,
            "stockout_risk_score": stockout_score,
            "overstock_risk_score": overstock_score,
            "risk_quadrant": quadrant,
            "recommended_action": action,
            "recommended_order_qty": rec_order_qty,
            "excess_units": excess_u,
            "lost_margin_risk": lost_margin_risk,
            "excess_capital_locked": excess_capital_locked,
            "inventory_value": inv_val,
            "total_financial_exposure": total_exposure,
            "unit_price": unit_price,
            "unit_cost": unit_cost,
        })

    risk_df = pd.DataFrame(risk_records)

    # Priority scoring based on exposure percentiles
    exposures = risk_df["total_financial_exposure"]
    q75 = float(exposures.quantile(0.75))
    q25 = float(exposures.quantile(0.25))

    risk_df["priority"] = risk_df.apply(
        lambda r: assign_action_priority(
            r["risk_quadrant"], r["total_financial_exposure"], q75, q25
        ),
        axis=1,
    )

    # Sort by priority and exposure
    p_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    risk_df["priority_rank"] = risk_df["priority"].map(p_order)
    risk_df = risk_df.sort_values(
        ["priority_rank", "total_financial_exposure"], ascending=[True, False]
    ).drop(columns=["priority_rank"]).reset_index(drop=True)

    # Summary Statistics
    quad_counts = risk_df["risk_quadrant"].value_counts().to_dict()
    portfolio_summary = {
        "total_skus": len(risk_df),
        "reorder_now_count": int(quad_counts.get("Reorder Now", 0)),
        "markdown_clear_count": int(quad_counts.get("Markdown / Clear", 0)),
        "watch_volatile_count": int(quad_counts.get("Watch / Volatile", 0)),
        "healthy_count": int(quad_counts.get("Healthy", 0)),
        "total_inventory_value_inr": round(float(risk_df["inventory_value"].sum()), 2),
        "total_lost_margin_risk_inr": round(float(risk_df["lost_margin_risk"].sum()), 2),
        "total_excess_capital_locked_inr": round(float(risk_df["excess_capital_locked"].sum()), 2),
        "total_reorder_units": int(risk_df["recommended_order_qty"].sum()),
        "total_excess_units_to_clear": int(risk_df["excess_units"].sum()),
    }

    return risk_df, portfolio_summary


def generate_inventory_risk_report(risk_df: pd.DataFrame, summary: dict) -> Path:
    """Generates the executive Inventory Risk Report in Markdown."""
    report_path = REPORTS_DIR / "inventory_risk_report.md"

    # Category breakdown table
    cat_summary = (
        risk_df.groupby("category")
        .agg(
            skus=("sku_id", "count"),
            reorder_skus=("risk_quadrant", lambda q: (q == "Reorder Now").sum()),
            markdown_skus=("risk_quadrant", lambda q: (q == "Markdown / Clear").sum()),
            inv_value=("inventory_value", "sum"),
            lost_margin=("lost_margin_risk", "sum"),
            excess_capital=("excess_capital_locked", "sum"),
        )
        .reset_index()
    )

    cat_rows = []
    for _, r in cat_summary.iterrows():
        cat_rows.append(
            f"| **{r['category']}** | {r['skus']} | {r['reorder_skus']} | {r['markdown_skus']} | "
            f"INR {r['inv_value']:,.2f} | INR {r['lost_margin']:,.2f} | INR {r['excess_capital']:,.2f} |"
        )
    cat_table_str = "\n".join(cat_rows)

    # Top 5 reorder candidates
    top_reorders = risk_df[risk_df["risk_quadrant"] == "Reorder Now"].head(5)
    reorder_rows = []
    for _, r in top_reorders.iterrows():
        reorder_rows.append(
            f"| **{r['sku_id']}** | {r['product_name']} | {r['category']} | {r['current_on_hand']} | "
            f"{r['reorder_point']} | **{r['recommended_order_qty']}** | INR {r['lost_margin_risk']:,.2f} |"
        )
    reorder_table_str = "\n".join(reorder_rows) if len(reorder_rows) > 0 else "| None | - | - | - | - | - | - |"

    # Top 5 markdown candidates
    top_markdowns = risk_df[risk_df["risk_quadrant"] == "Markdown / Clear"].head(5)
    markdown_rows = []
    for _, r in top_markdowns.iterrows():
        markdown_rows.append(
            f"| **{r['sku_id']}** | {r['product_name']} | {r['category']} | {r['current_on_hand']} | "
            f"{r['weeks_of_cover']} wks | **{r['excess_units']}** | INR {r['excess_capital_locked']:,.2f} |"
        )
    markdown_table_str = "\n".join(markdown_rows) if len(markdown_rows) > 0 else "| None | - | - | - | - | - | - |"

    report_content = f"""# Inventory Risk & Capital Optimization Report: Project FORESIGHT

> **Document Status**: Complete / Executive Ready  
> **Prepared For**: Head of Operations & Finance Lead  
> **Prepared By**: Data Science & Analytics Team  
> **Effective Horizon**: 12-Week Forward Operational Window  

---

## 1. Executive Summary & Headline Financial Impact

Through integration of ML weekly demand forecasts with warehouse stock ledgers, the **FORESIGHT Risk Engine** evaluated all 300 SKUs against dynamic safety stock thresholds, weeks of cover, and supplier lead times:

* **Total Current Warehouse Inventory Value**: **INR {summary['total_inventory_value_inr']:,.2f}**
* **Total Gross Margin at Risk (Stockout Threat)**: **INR {summary['total_lost_margin_risk_inr']:,.2f}**
* **Total Working Capital Locked in Excess Stock**: **INR {summary['total_excess_capital_locked_inr']:,.2f}**
* **Total Capital Optimization Opportunity**: **INR {summary['total_lost_margin_risk_inr'] + summary['total_excess_capital_locked_inr']:,.2f}**

> [!IMPORTANT]
> **Data Provenance Disclosure**:
> {DATASET_DISCLAIMER}

---

## 2. 2D Decision Grid Distribution

Every SKU is assigned to exactly one operational quadrant based on its normalized stockout score ($Y$-axis) and overstock score ($X$-axis):

| Decision Quadrant | Operational Meaning | SKU Count | Share of Portfolio | Primary Action |
| :--- | :--- | :--- | :--- | :--- |
| **Reorder Now** | High Stockout, Low Overstock | **{summary['reorder_now_count']} SKUs** | **{summary['reorder_now_count']/summary['total_skus']*100:.1f}%** | Urgent purchase order to prevent stockout |
| **Markdown / Clear** | Low Stockout, High Overstock | **{summary['markdown_clear_count']} SKUs** | **{summary['markdown_clear_count']/summary['total_skus']*100:.1f}%** | Targeted promotion to liquidate locked capital |
| **Watch / Volatile** | High on Both / Erratic Demand | **{summary['watch_volatile_count']} SKUs** | **{summary['watch_volatile_count']/summary['total_skus']*100:.1f}%** | Manual audit; investigate supplier jitter |
| **Healthy** | Low Stockout, Low Overstock | **{summary['healthy_count']} SKUs** | **{summary['healthy_count']/summary['total_skus']*100:.1f}%** | Routine monitoring; standard replenishment |

---

## 3. Category-Level Risk & Financial Exposure

| Product Category | Total SKUs | Reorder Candidates | Markdown Candidates | Total Inventory Value | Lost Margin at Risk | Locked Excess Capital |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{cat_table_str}

---

## 4. Priority Operational Action Lists

### A. Top 5 Urgent Reorder Candidates (High Lost-Margin Risk)
| SKU ID | Product Name | Category | On-Hand | ROP | Rec. Order Qty | Lost Margin at Risk (INR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{reorder_table_str}

### B. Top 5 Markdown / Clearance Candidates (High Excess Working Capital)
| SKU ID | Product Name | Category | On-Hand | Weeks of Cover | Excess Units | Capital Locked (INR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{markdown_table_str}

---

## 5. Mathematical & Policy Methodology

1. **Safety Stock Formulation**:
   $$SS = \\lceil Z \\cdot \\sigma_{{\\text{{weekly\\_error}}}} \\cdot \\sqrt{{\\text{{Lead Time (Weeks)}}}} \\rceil$$
   Calibrated using SKU-specific forecast residual variance from backtest evaluation and target service levels ($Z \\in [1.28, 2.05]$).
2. **Reorder Point (ROP)**:
   $$\\text{{ROP}} = \\lceil \\hat{{d}}_{{\\text{{weekly}}}} \\cdot \\text{{Lead Time (Weeks)}} + SS \\rceil$$
   Driven strictly by forward model predictions, eliminating reliance on lagging historical averages.
3. **Lost Margin Risk Calculation**:
   $$\\text{{Lost Margin Risk (INR)}} = \\max(0, \\text{{LTD}} - \\text{{Effective Available}}) \\times (\\text{{Price}} - \\text{{Cost}})$$
   Reflects true bottom-line profitability lost rather than gross revenue.
4. **Excess Capital Locked Calculation**:
   $$\\text{{Excess Capital Locked (INR)}} = \\max(0, \\text{{On-Hand}} - \\text{{Target Stock}}) \\times \\text{{Cost}}$$

---

## 6. Assumptions & Limitations

1. **Ordering Cost Constant**: Assumed standard procurement administration cost $S = \\text{{INR }} 500$ per PO line item for EOQ sizing.
2. **Holding Cost Rate**: Standard 20% annual cost of capital and warehousing applied to unit cost.
3. **Supplier Lead Time Rigidity**: Lead times are treated as deterministic; real supply chains experience shipping variance requiring continuous buffer recalibration.

---
*Report generated automatically by Project FORESIGHT Inventory Intelligence Engine.*
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info("Saved Inventory Risk Report to %s", report_path)
    return report_path


def main():
    logger.info("=" * 60)
    logger.info("STARTING PHASE 5: 2D INVENTORY RISK ENGINE & RUPEE IMPACT")
    logger.info("=" * 60)

    # 1. Load Required Artifacts
    sku_master = pd.read_csv(SYNTHETIC_DATA_DIR / "sku_master.csv")
    inv_snapshots = pd.read_csv(SYNTHETIC_DATA_DIR / "inventory_snapshots.csv")
    forecast_results = pd.read_csv(FORECASTS_DIR / "forecast_results.csv")
    backtest_results = pd.read_csv(REPORTS_DIR / "backtest_results.csv")

    # 2. Build Risk Register
    risk_df, summary = build_inventory_risk_register(
        sku_master=sku_master,
        inventory_snapshots=inv_snapshots,
        forecast_results=forecast_results,
        backtest_results=backtest_results,
    )

    # 3. Save Risk Register
    RISK_DIR.mkdir(parents=True, exist_ok=True)
    out_register_path = RISK_DIR / "risk_register.csv"
    risk_df.to_csv(out_register_path, index=False)
    logger.info("Saved complete risk register to %s", out_register_path)

    # 4. Generate 2D Decision Grid Chart
    plot_risk_decision_grid(risk_df)

    # 5. Generate Formal Executive Report
    report_file = generate_inventory_risk_report(risk_df, summary)

    # 6. Print Verification Summary
    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETION AUDIT & INVENTORY RISK SUMMARY")
    print("=" * 60)
    print(f"1. TOTAL SKUS SCORED: {summary['total_skus']} active SKUs (100% catalog coverage)")
    print("\n2. 2D DECISION GRID QUADRANT COUNTS:")
    print(f"   - Reorder Now (High Stockout, Low Overstock): {summary['reorder_now_count']} SKUs ({summary['reorder_now_count']/summary['total_skus']*100:.1f}%)")
    print(f"   - Markdown / Clear (Low Stockout, High Overstock): {summary['markdown_clear_count']} SKUs ({summary['markdown_clear_count']/summary['total_skus']*100:.1f}%)")
    print(f"   - Watch / Volatile (High on Both): {summary['watch_volatile_count']} SKUs ({summary['watch_volatile_count']/summary['total_skus']*100:.1f}%)")
    print(f"   - Healthy (Low on Both): {summary['healthy_count']} SKUs ({summary['healthy_count']/summary['total_skus']*100:.1f}%)")

    print("\n3. PORTFOLIO FINANCIAL IMPACT METRICS:")
    print(f"   - Current Warehouse Inventory Value: INR {summary['total_inventory_value_inr']:,.2f}")
    print(f"   - Total Lost Margin at Risk (Stockouts): INR {summary['total_lost_margin_risk_inr']:,.2f}")
    print(f"   - Total Working Capital Locked (Overstock): INR {summary['total_excess_capital_locked_inr']:,.2f}")
    print(f"   - Total Recommended Reorder Units: {summary['total_reorder_units']:,} units")
    print(f"   - Total Excess Units to Liquidate: {summary['total_excess_units_to_clear']:,} units")

    print("\n4. TOP 5 URGENT REORDER CANDIDATES:")
    top_reorders = risk_df[risk_df["risk_quadrant"] == "Reorder Now"].head(5)
    for _, r in top_reorders.iterrows():
        print(f"   - {r['sku_id']} ({r['product_name']}): On-Hand={r['current_on_hand']}, ROP={r['reorder_point']}, OrderQty={r['recommended_order_qty']}, LostMargin=INR {r['lost_margin_risk']:,.2f}")

    print("\n5. TOP 5 MARKDOWN / CLEARANCE CANDIDATES:")
    top_markdowns = risk_df[risk_df["risk_quadrant"] == "Markdown / Clear"].head(5)
    for _, r in top_markdowns.iterrows():
        print(f"   - {r['sku_id']} ({r['product_name']}): On-Hand={r['current_on_hand']}, WOC={r['weeks_of_cover']} wks, ExcessUnits={r['excess_units']}, LockedCapital=INR {r['excess_capital_locked']:,.2f}")

    print("\n6. FINANCIAL INTEGRITY CHECKS:")
    print(f"   - Negative lost margin risk records: {(risk_df['lost_margin_risk'] < 0).sum()}")
    print(f"   - Negative excess capital locked records: {(risk_df['excess_capital_locked'] < 0).sum()}")
    print(f"   - Negative recommended order qty records: {(risk_df['recommended_order_qty'] < 0).sum()}")
    print(f"   - Stockout risk scores outside [0, 1]: {((risk_df['stockout_risk_score'] < 0) | (risk_df['stockout_risk_score'] > 1)).sum()}")
    print(f"   - Overstock risk scores outside [0, 1]: {((risk_df['overstock_risk_score'] < 0) | (risk_df['overstock_risk_score'] > 1)).sum()}")
    print(f"   - Duplicate SKU records in risk register: {risk_df['sku_id'].duplicated().sum()}")

    print("\n7. OUTPUT ARTIFACTS GENERATED:")
    print(f"   - Risk Register: outputs/risk/risk_register.csv")
    print(f"   - Decision Grid Chart: outputs/charts/risk_decision_grid.png")
    print(f"   - Risk Report: outputs/reports/inventory_risk_report.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
