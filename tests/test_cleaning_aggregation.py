import pytest
import pandas as pd
import numpy as np

from src.data.cleaning import clean_datasets
from src.data.aggregation import aggregate_daily_to_weekly
from src.data.synthetic_generator import generate_sku_master, generate_calendar, simulate_demand_and_inventory


@pytest.fixture(scope="module")
def mini_clean_pipeline():
    rng = np.random.default_rng(42)
    sku_df = generate_sku_master(n_skus=10, rng=rng)
    cal_df = generate_calendar(start_date="2024-01-01", end_date="2024-02-28")
    sales_df, inv_df = simulate_demand_and_inventory(sku_df, cal_df, rng=rng)
    return sku_df, cal_df, sales_df, inv_df


def test_clean_datasets_preserves_valid(mini_clean_pipeline):
    sku_df, cal_df, sales_df, inv_df = mini_clean_pipeline
    sku_c, cal_c, sales_c, inv_c, audit_df = clean_datasets(sku_df, cal_df, sales_df, inv_df, save_processed=False)
    assert len(sku_c) == len(sku_df)
    assert len(cal_c) == len(cal_df)
    assert len(sales_c) == len(sales_df)
    assert len(inv_c) == len(inv_df)
    assert "No corrective cleaning required" in audit_df.iloc[0]["reason"]


def test_clean_datasets_removes_duplicates(mini_clean_pipeline):
    sku_df, cal_df, sales_df, inv_df = mini_clean_pipeline
    # Inject deliberate duplicate
    dup_sales = pd.concat([sales_df, sales_df.iloc[[0]]], ignore_index=True)
    sku_c, cal_c, sales_c, inv_c, audit_df = clean_datasets(sku_df, cal_df, dup_sales, inv_df, save_processed=False)
    assert len(sales_c) == len(sales_df)
    assert any("Duplicate (date, sku_id) transactions" in r["issue"] for _, r in audit_df.iterrows())


def test_aggregation_reconciliation(mini_clean_pipeline):
    sku_df, cal_df, sales_df, _ = mini_clean_pipeline
    weekly, recon = aggregate_daily_to_weekly(sales_df, sku_df, cal_df, save_csv=False)
    assert recon["units_discrepancy"] == 0
    assert recon["revenue_discrepancy"] == 0
    assert (weekly["weekly_units_sold"] >= 0).all()
    assert (weekly["weekly_revenue"] >= 0).all()
    assert (weekly["promotion_days"].between(0, 7)).all()
    assert (weekly["stockout_days"].between(0, 7)).all()
    assert (weekly["stockout_rate"].between(0.0, 1.0)).all()
