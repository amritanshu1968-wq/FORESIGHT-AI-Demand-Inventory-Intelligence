import pytest
import pandas as pd
import numpy as np

from src.data.synthetic_generator import generate_sku_master, generate_calendar, simulate_demand_and_inventory
from src.data.validation import (
    validate_primary_keys,
    validate_foreign_keys,
    validate_non_negativity_and_ranges,
    validate_inventory_conservation,
    DataValidationError,
)


@pytest.fixture(scope="module")
def mini_dataset():
    """Generates a small fast test dataset for quick unit testing."""
    rng = np.random.default_rng(42)
    sku_df = generate_sku_master(n_skus=20, rng=rng)
    cal_df = generate_calendar(start_date="2024-01-01", end_date="2024-03-31")
    sales_df, inv_df = simulate_demand_and_inventory(sku_df, cal_df, rng=rng)
    return sku_df, cal_df, sales_df, inv_df


def test_sku_master_integrity(mini_dataset):
    sku_df, _, _, _ = mini_dataset
    assert len(sku_df) == 20
    assert sku_df["sku_id"].nunique() == 20
    assert (sku_df["unit_cost"] < sku_df["unit_price"]).all()
    assert np.allclose(sku_df["unit_price"] - sku_df["unit_cost"], sku_df["margin_per_unit"], atol=0.01)


def test_calendar_integrity(mini_dataset):
    _, cal_df, _, _ = mini_dataset
    assert len(cal_df) == 91  # 2024 is leap year, Jan 31 + Feb 29 + Mar 31 = 91 days
    assert cal_df["date"].nunique() == 91
    assert set(cal_df["day_of_week"].unique()).issubset(set(range(1, 8)))


def test_inventory_conservation_strict(mini_dataset):
    _, _, sales_df, inv_df = mini_dataset
    # Equation: Closing = Opening + Received - Sold
    expected_closing = inv_df["opening_stock"] + inv_df["stock_received"] - inv_df["units_sold"]
    assert (inv_df["closing_stock"] == expected_closing).all()

    # Physical constraint: Sold <= Opening + Received
    assert (inv_df["units_sold"] <= (inv_df["opening_stock"] + inv_df["stock_received"])).all()

    # No negative numbers
    assert (inv_df["opening_stock"] >= 0).all()
    assert (inv_df["closing_stock"] >= 0).all()
    assert (inv_df["units_sold"] >= 0).all()


def test_temporal_continuity(mini_dataset):
    _, _, _, inv_df = mini_dataset
    sorted_inv = inv_df.sort_values(["sku_id", "snapshot_date"]).copy()
    sorted_inv["prev_closing"] = sorted_inv.groupby("sku_id")["closing_stock"].shift(1)
    continuity_slice = sorted_inv.dropna(subset=["prev_closing"])
    assert (continuity_slice["opening_stock"] == continuity_slice["prev_closing"]).all()


def test_validation_detects_deliberate_error(mini_dataset):
    sku_df, cal_df, sales_df, inv_df = mini_dataset
    # Corrupt a row in inventory
    bad_inv = inv_df.copy()
    bad_inv.loc[0, "closing_stock"] += 50
    with pytest.raises(DataValidationError):
        validate_inventory_conservation(sales_df, bad_inv)
