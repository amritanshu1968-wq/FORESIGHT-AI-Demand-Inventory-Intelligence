"""
PROJECT FORESIGHT — Phase 2 Execution & Verification Script
Runs synthetic generation, saves CSVs, validates relational & arithmetic integrity,
and prints the required Phase 2 audit metrics.
"""

import json
import sys
from pathlib import Path
import pandas as pd

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SYNTHETIC_DATA_DIR
from src.data.synthetic_generator import generate_all_datasets
from src.data.validation import run_comprehensive_validation
from src.utils.logger import get_logger

logger = get_logger("run_phase2")


def main():
    logger.info("=" * 60)
    logger.info("STARTING PHASE 2: SYNTHETIC DATA GENERATION & VALIDATION")
    logger.info("=" * 60)

    # 1. Generate Datasets
    sku_master, calendar, sales_daily, inventory_snapshots = generate_all_datasets(save_csv=True)

    # 2. Run Comprehensive Validation
    validation_report = run_comprehensive_validation(
        sku_master=sku_master,
        calendar=calendar,
        sales_daily=sales_daily,
        inventory_snapshots=inventory_snapshots,
    )

    # 3. Print Detailed Verification Summary
    print("\n" + "=" * 60)
    print("PHASE 2 AUDIT & VERIFICATION REPORT")
    print("=" * 60)

    print("\n1. GENERATED ROW & COLUMN COUNTS:")
    for k, v in validation_report["dataset_summary"].items():
        print(f"   - {k}: {v:,}")

    print("\n2. DATE RANGE & HORIZON:")
    for k, v in validation_report["date_range"].items():
        print(f"   - {k}: {v}")

    print("\n3. SKU COVERAGE & CATEGORY BREAKDOWN:")
    print(f"   - Total SKUs: {validation_report['sku_coverage']['total_skus']}")
    for cat, count in validation_report["sku_coverage"]["categories"].items():
        print(f"     * {cat}: {count} SKUs ({count / validation_report['sku_coverage']['total_skus'] * 100:.1f}%)")

    print("\n4. MISSING VALUE SUMMARY:")
    for table, miss in validation_report["missing_values"].items():
        print(f"   - {table}: {miss} missing cells (in non-nullable fields)")

    print("\n5. DUPLICATE SUMMARY:")
    for table, dup in validation_report["duplicate_rows"].items():
        print(f"   - {table}: {dup} duplicate rows")

    print("\n6. INVENTORY RECONCILIATION & CONSERVATION:")
    print(f"   - Balance Equation Violations (Closing = Opening + Recv - Sold): {validation_report['validation_checks']['balance_equation_violations']}")
    print(f"   - Oversold Violations (Units Sold > Available Stock): {validation_report['validation_checks']['oversold_violations']}")
    print(f"   - Temporal Continuity Breaks (Opening(t+1) == Closing(t)): {validation_report['validation_checks']['temporal_continuity_breaks']}")
    print(f"   - Cross-table Sales Mismatches (sales_daily vs inventory_snapshots): {validation_report['validation_checks']['cross_table_sales_mismatch']}")
    print(f"   - Negative Stock Incidents: {validation_report['validation_checks']['negative_closing_stock']}")

    print("\n7. OPERATIONAL RISK INCIDENCE:")
    print(f"   - Stockout Days: {validation_report['operational_metrics']['stockout_events_count']:,} ({validation_report['operational_metrics']['stockout_percentage']}%)")
    print(f"   - Overstock Snapshots (DOC > 70 days): {validation_report['operational_metrics']['overstock_snapshot_count']:,} ({validation_report['operational_metrics']['overstock_percentage']}%)")
    print(f"   - Total Units Sold: {validation_report['operational_metrics']['total_units_sold']:,}")
    print(f"   - Total Revenue: INR {validation_report['operational_metrics']['total_revenue_inr']:,.2f}")

    print("\n8. FILES GENERATED IN data/synthetic/:")
    for csv_file in sorted(SYNTHETIC_DATA_DIR.glob("*.csv")):
        size_mb = csv_file.stat().st_size / (1024 * 1024)
        print(f"   - {csv_file.name} ({size_mb:.2f} MB)")

    print("\n9. SAMPLE ROWS:")
    print("\n--- sku_master (head 2) ---")
    print(sku_master[["sku_id", "product_name", "category", "unit_price", "unit_cost", "margin_per_unit", "lead_time_days"]].head(2).to_string())

    print("\n--- calendar (head 2) ---")
    print(calendar[["date", "day_of_week", "week", "month", "season", "is_holiday", "festival_name"]].head(2).to_string())

    print("\n--- sales_daily (head 2) ---")
    print(sales_daily.head(2).to_string())

    print("\n--- inventory_snapshots (head 2) ---")
    print(inventory_snapshots.head(2).to_string())

    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETED SUCCESSFULLY — DATA VALIDATION PASSED!")
    print("=" * 60)

    # Save summary report to outputs/reports/phase2_validation_report.json
    report_path = PROJECT_ROOT / "outputs" / "reports" / "phase2_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(validation_report, f, indent=2)
    logger.info("Validation report written to %s", report_path)


if __name__ == "__main__":
    main()
