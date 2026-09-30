"""
PROJECT FORESIGHT — Synthetic Retail Data Generator
Generates realistic, internally consistent retail data for NorthBay Living.

Tables Generated:
1. sku_master.csv (Product catalog dimension)
2. calendar.csv (Calendar & Indian retail festival dimension)
3. sales_daily.csv (Daily sales transactions fact table)
4. inventory_snapshots.csv (Daily stock ledger fact table)

DISCLAIMER:
All data generated is SYNTHETIC retail data created specifically for project development
and demonstration. It does not represent actual confidential business results of
NorthBay Living or Zidio Development.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from config import (
    DATASET_DISCLAIMER,
    END_DATE,
    N_SKUS,
    PRODUCT_TAXONOMY,
    RANDOM_SEED,
    START_DATE,
    SYNTHETIC_DATA_DIR,
)
from src.utils.logger import get_logger

logger = get_logger("synthetic_generator")


def generate_sku_master(n_skus: int = N_SKUS, rng: np.random.Generator = None) -> pd.DataFrame:
    """Generates the SKU catalog dimension table."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    categories = list(PRODUCT_TAXONOMY.keys())
    sku_rows = []

    # Assign demand profiles: 20% Fast, 30% Medium, 35% Slow, 15% Intermittent
    n_fast = int(0.20 * n_skus)
    n_med = int(0.30 * n_skus)
    n_slow = int(0.35 * n_skus)
    n_intermittent = n_skus - n_fast - n_med - n_slow

    profiles = (
        ["fast"] * n_fast
        + ["medium"] * n_med
        + ["slow"] * n_slow
        + ["intermittent"] * n_intermittent
    )
    rng.shuffle(profiles)

    # Assign inventory behavior scenarios:
    # 'healthy': 64%, 'stockout_prone': 12%, 'overstock_prone': 12%, 'volatile': 12%
    scenarios = (
        ["healthy"] * int(0.64 * n_skus)
        + ["stockout_prone"] * int(0.12 * n_skus)
        + ["overstock_prone"] * int(0.12 * n_skus)
        + ["volatile"] * (n_skus - int(0.64 * n_skus) - 2 * int(0.12 * n_skus))
    )
    rng.shuffle(scenarios)

    brands = ["NorthBay Studio", "NorthBay Basics", "NorthBay Reserve"]
    brand_weights = [0.45, 0.40, 0.15]

    for i in range(n_skus):
        sku_id = f"SKU-{i+1:04d}"
        category = categories[i % len(categories)]
        cat_meta = PRODUCT_TAXONOMY[category]
        subcategory = rng.choice(cat_meta["subcategories"])
        brand = rng.choice(brands, p=brand_weights)

        # Price and margin architecture
        p_min, p_max = cat_meta["base_price_range"]
        list_price = round(float(rng.uniform(p_min, p_max)), 2)
        # Margin typically 35% to 60%
        margin_pct = float(rng.uniform(0.38, 0.58))
        unit_cost = round(list_price * (1.0 - margin_pct), 2)
        margin_per_unit = round(list_price - unit_cost, 2)

        # Lead times
        lt_min, lt_max = cat_meta["lead_time_days_range"]
        lead_time_days = int(rng.integers(lt_min, lt_max + 1))

        # Supplier, MOQ, Target Service Level
        supplier_id = f"SUP-{rng.integers(1, 16):02d}"
        moq = int(rng.choice([15, 25, 50, 75, 100]))
        target_service_level = round(float(rng.choice([0.90, 0.95, 0.98], p=[0.2, 0.6, 0.2])), 2)

        # Launch date
        launch_offset_days = int(rng.integers(0, 365))
        launch_date = (datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=launch_offset_days)).strftime("%Y-%m-%d")

        shelf_life_days = int(rng.choice([365, 730, 1095, 1825]))
        status = "Active" if rng.random() > 0.05 else "Phasing Out"
        product_name = f"{brand} {subcategory} #{i+1:03d}"

        sku_rows.append({
            "sku_id": sku_id,
            "product_name": product_name,
            "category": category,
            "subcategory": subcategory,
            "brand": brand,
            "unit_price": list_price,
            "unit_cost": unit_cost,
            "margin_per_unit": margin_per_unit,
            "lead_time_days": lead_time_days,
            "minimum_order_qty": moq,
            "target_service_level": target_service_level,
            "shelf_life_days": shelf_life_days,
            "supplier_id": supplier_id,
            "status": status,
            "launch_date": launch_date,
            "demand_profile": profiles[i],
            "inventory_scenario": scenarios[i],
        })

    df = pd.DataFrame(sku_rows)
    return df


def generate_calendar(start_date: str = START_DATE, end_date: str = END_DATE) -> pd.DataFrame:
    """Generates the calendar dimension table with Indian retail holidays and festival seasons."""
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    records = []

    for d in date_range:
        date_str = d.strftime("%Y-%m-%d")
        month = d.month
        day = d.day
        year = d.year
        iso_year, iso_week, day_of_week = d.isocalendar()

        # Season
        if month in [12, 1, 2]:
            season = "Winter"
        elif month in [3, 4]:
            season = "Spring"
        elif month in [5, 6]:
            season = "Summer"
        elif month in [7, 8, 9]:
            season = "Monsoon"
        else:
            season = "Festive/Autumn"

        # Indian Public Holidays & Festival windows (approximate calendar dates)
        is_holiday = False
        holiday_name = None
        festival = False
        festival_name = None
        promo_event = None

        # Fixed holidays
        if month == 1 and day == 26:
            is_holiday = True
            holiday_name = "Republic Day"
            promo_event = "Republic Day Sale"
        elif month == 8 and day == 15:
            is_holiday = True
            holiday_name = "Independence Day"
            promo_event = "Independence Day Freedom Sale"
        elif month == 10 and day == 2:
            is_holiday = True
            holiday_name = "Gandhi Jayanti"
        elif month == 12 and day == 25:
            is_holiday = True
            holiday_name = "Christmas"
            promo_event = "Christmas & Year End Mega Sale"
        elif month == 1 and day == 1:
            is_holiday = True
            holiday_name = "New Year's Day"
            promo_event = "New Year Bash"

        # Variable festivals by year
        # Holi: ~March
        holi_dates = {2023: (3, 8), 2024: (3, 25), 2025: (3, 14)}
        if (month, day) == holi_dates.get(year):
            is_holiday = True
            holiday_name = "Holi"
            festival = True
            festival_name = "Holi Festival"
            promo_event = "Holi Colours Mega Sale"

        # Diwali Season: usually mid-Oct to mid-Nov (ISO weeks 41-46)
        diwali_main_dates = {2023: (11, 12), 2024: (11, 1), 2025: (10, 20)}
        d_m, d_d = diwali_main_dates.get(year, (11, 5))
        d_diwali = datetime(year, d_m, d_d)
        diff_days = (d.to_pydatetime() - d_diwali).days

        # Peak Diwali shopping window: 14 days before to 3 days after
        if -14 <= diff_days <= 3:
            festival = True
            festival_name = "Diwali Festive Season"
            promo_event = "Great Indian Festive Home Sale"
            if diff_days == 0:
                is_holiday = True
                holiday_name = "Diwali (Deepavali)"
            elif diff_days == -2:
                holiday_name = "Dhanteras"

        # Summer Home Refresh Sale (May)
        if month == 5 and 15 <= day <= 22:
            promo_event = "Summer Home Refresh Sale"

        records.append({
            "date": date_str,
            "day_of_week": int(day_of_week),  # 1=Monday, 7=Sunday
            "week": int(iso_week),
            "month": int(month),
            "quarter": int((month - 1) // 3 + 1),
            "year": int(year),
            "is_weekend": int(day_of_week in [6, 7]),
            "is_holiday": int(is_holiday),
            "holiday_name": holiday_name,
            "festival": int(festival),
            "festival_name": festival_name,
            "season": season,
            "is_month_start": int(d.is_month_start),
            "is_month_end": int(d.is_month_end),
            "promo_event": promo_event,
        })

    return pd.DataFrame(records)


def simulate_demand_and_inventory(
    sku_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    rng: np.random.Generator = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Co-simulates daily customer demand, sales fulfillment, and inventory ledgers.
    Maintains strict physical conservation:
    Closing(t) = Opening(t) + Received(t) - Sold(t)
    Opening(t+1) = Closing(t)
    Units Sold(t) <= Opening(t) + Received(t)
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    dates = cal_df["date"].values
    n_days = len(dates)
    day_indices = np.arange(n_days)

    # Pre-extract calendar arrays
    dow_arr = cal_df["day_of_week"].values
    is_weekend_arr = cal_df["is_weekend"].values
    is_holiday_arr = cal_df["is_holiday"].values
    festival_arr = cal_df["festival"].values
    promo_event_mask = cal_df["promo_event"].notna().values

    sales_rows: List[dict] = []
    inventory_rows: List[dict] = []

    # Iterate over each SKU
    for _, sku_row in sku_df.iterrows():
        sku_id = sku_row["sku_id"]
        category = sku_row["category"]
        list_price = sku_row["unit_price"]
        unit_cost = sku_row["unit_cost"]
        lead_time = int(sku_row["lead_time_days"])
        moq = int(sku_row["minimum_order_qty"])
        profile = sku_row["demand_profile"]
        scenario = sku_row["inventory_scenario"]

        # Base daily demand scale
        if profile == "fast":
            base_demand = float(rng.uniform(14.0, 32.0))
        elif profile == "medium":
            base_demand = float(rng.uniform(5.0, 13.0))
        elif profile == "slow":
            base_demand = float(rng.uniform(0.7, 3.5))
        else:  # intermittent
            base_demand = float(rng.uniform(0.2, 1.2))

        # Long term growth trend
        trend_rate = float(rng.normal(0.08, 0.03))  # 8% annual growth

        # Sinusoidal annual seasonality
        phase = float(rng.uniform(0, 2 * np.pi))
        seasonality = 1.0 + 0.20 * np.sin(2 * np.pi * day_indices / 365.25 + phase)

        # Weekend uplift (~25% retail surge)
        weekend_mult = np.where(is_weekend_arr == 1, 1.25, 1.0)

        # Festival multiplier (Oct-Nov Diwali surge)
        cat_festive_boost = {
            "Home Furnishings": 1.70,
            "Décor & Lighting": 1.85,
            "Kitchen & Dining": 1.50,
            "Small Appliances": 1.65,
        }.get(category, 1.4)
        festival_mult = np.where(festival_arr == 1, cat_festive_boost, 1.0)

        # Promotional schedule: SKU participates in calendar promo events + some random flash promos
        sku_promo_flag = (
            promo_event_mask & (rng.random(n_days) < 0.65)
        ) | (rng.random(n_days) < 0.03)

        # Discounts when on promo
        discount_pcts = np.where(
            sku_promo_flag,
            rng.choice([0.10, 0.15, 0.20, 0.25, 0.30], size=n_days, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
            0.0,
        )
        promo_elasticity = 1.0 + 1.8 * discount_pcts

        # Unconstrained expected demand mean
        mu_demand = (
            base_demand
            * (1.0 + trend_rate * day_indices / 365.25)
            * seasonality
            * weekend_mult
            * festival_mult
            * promo_elasticity
        )

        # Generate unconstrained demand realizations
        if profile == "intermittent":
            # Zero-inflated demand
            prob_purchase = np.clip(mu_demand * 0.18, 0.05, 0.40)
            has_demand = rng.random(n_days) < prob_purchase
            unconstrained_demand = np.where(
                has_demand,
                rng.integers(1, 5, size=n_days),
                0,
            )
        else:
            # Poisson + normal variability
            poisson_part = rng.poisson(np.maximum(mu_demand * 0.75, 0.1))
            gaussian_noise = np.round(rng.normal(0, np.maximum(mu_demand * 0.15, 0.1))).astype(int)
            unconstrained_demand = np.clip(poisson_part + gaussian_noise, 0, None)

        # Inventory policy parameters
        avg_daily_demand = max(float(np.mean(unconstrained_demand)), 0.1)
        lead_time_demand = avg_daily_demand * lead_time
        sigma_daily = max(float(np.std(unconstrained_demand)), 0.5)
        safety_stock = int(np.ceil(1.645 * sigma_daily * np.sqrt(lead_time)))
        reorder_point = int(np.ceil(lead_time_demand + safety_stock))
        # Economic Order Quantity (annual demand = daily * 365, order_cost=500, holding=unit_cost*0.2)
        annual_demand = avg_daily_demand * 365.0
        eoq = int(np.ceil(np.sqrt((2.0 * annual_demand * 500.0) / max(unit_cost * 0.20, 1.0))))
        order_batch_size = max(moq, eoq)

        # Initial opening stock
        if scenario == "stockout_prone":
            opening_stock = int(avg_daily_demand * lead_time * 0.6)  # deliberately lean
        elif scenario == "overstock_prone":
            opening_stock = int(avg_daily_demand * 90)  # ~3 months on hand
        else:
            opening_stock = int(avg_daily_demand * 30 + safety_stock)

        # Pipeline of pending purchase orders: dict of arrival_day_index -> quantity
        in_flight_orders: Dict[int, int] = {}

        for t in range(n_days):
            date_val = dates[t]

            # 1. Receive incoming purchase orders arriving today
            stock_received = in_flight_orders.pop(t, 0)

            # Available stock
            available_stock = opening_stock + stock_received

            # 2. Demand & Fulfill
            dem = int(unconstrained_demand[t])
            if dem > available_stock:
                units_sold = available_stock
                stockout = 1
            else:
                units_sold = dem
                stockout = 0

            # 3. Closing Stock (Strict arithmetic conservation)
            closing_stock = available_stock - units_sold

            # 4. Pricing & Revenue
            disc = float(discount_pcts[t])
            realized_price = round(list_price * (1.0 - disc), 2)
            revenue = round(units_sold * realized_price, 2)
            is_promo = int(sku_promo_flag[t])

            sales_rows.append({
                "date": date_val,
                "sku_id": sku_id,
                "units_sold": units_sold,
                "unit_price": realized_price,
                "revenue": revenue,
                "promotion_flag": is_promo,
                "discount_pct": disc,
                "channel": "Online D2C Store",
                "stockout_flag": stockout,
            })

            # 5. Inventory Snapshot Metrics
            total_on_order = sum(in_flight_orders.values())
            days_of_cover = round(closing_stock / avg_daily_demand, 1)
            inventory_val = round(closing_stock * unit_cost, 2)

            inventory_rows.append({
                "snapshot_date": date_val,
                "sku_id": sku_id,
                "opening_stock": opening_stock,
                "stock_received": stock_received,
                "units_sold": units_sold,
                "closing_stock": closing_stock,
                "on_order_units": total_on_order,
                "lead_time_days": lead_time,
                "reorder_point": reorder_point,
                "safety_stock": safety_stock,
                "days_of_cover": days_of_cover,
                "stockout_flag": stockout,
                "inventory_value": inventory_val,
            })

            # 6. Replenishment policy execution
            inventory_position = closing_stock + total_on_order

            if scenario == "stockout_prone":
                # Supplier delays or sporadic missed replenishment orders
                if inventory_position <= int(reorder_point * 0.7) and len(in_flight_orders) == 0:
                    delay_jitter = lead_time + int(rng.integers(5, 18))  # severe supplier delay
                    arrival_t = t + delay_jitter
                    in_flight_orders[arrival_t] = order_batch_size
            elif scenario == "overstock_prone":
                # Over-ordering batches regardless of high cover
                if inventory_position <= reorder_point * 2.5 and len(in_flight_orders) < 2:
                    arrival_t = t + lead_time
                    in_flight_orders[arrival_t] = int(order_batch_size * 2.2)  # oversized batches
            elif scenario == "volatile":
                # Erratic replenishment
                if inventory_position <= reorder_point and len(in_flight_orders) == 0:
                    arrival_t = t + lead_time + int(rng.integers(-2, 7))
                    in_flight_orders[arrival_t] = order_batch_size
            else:  # healthy standard (ROP, Q)
                if inventory_position <= reorder_point and len(in_flight_orders) == 0:
                    arrival_t = t + lead_time + int(rng.integers(-1, 3))
                    in_flight_orders[arrival_t] = order_batch_size

            # 7. Next day opening stock continuity
            opening_stock = closing_stock

    sales_df = pd.DataFrame(sales_rows)
    inv_df = pd.DataFrame(inventory_rows)

    return sales_df, inv_df


def generate_all_datasets(save_csv: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates all 4 synthetic datasets and saves them to data/synthetic/."""
    logger.info("Initializing synthetic data generation with seed %d...", RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # 1. SKU Master
    logger.info("Generating SKU Master (%d SKUs)...", N_SKUS)
    sku_master = generate_sku_master(n_skus=N_SKUS, rng=rng)

    # 2. Calendar
    logger.info("Generating Calendar dimension (%s to %s)...", START_DATE, END_DATE)
    calendar = generate_calendar(start_date=START_DATE, end_date=END_DATE)

    # 3. Sales Daily & Inventory Snapshots (Co-simulated)
    logger.info("Simulating daily sales and inventory ledgers across %d SKUs and %d days...", N_SKUS, len(calendar))
    sales_daily, inventory_snapshots = simulate_demand_and_inventory(sku_master, calendar, rng=rng)

    if save_csv:
        logger.info("Saving generated CSV datasets to %s...", SYNTHETIC_DATA_DIR)
        sku_master.to_csv(SYNTHETIC_DATA_DIR / "sku_master.csv", index=False)
        calendar.to_csv(SYNTHETIC_DATA_DIR / "calendar.csv", index=False)
        sales_daily.to_csv(SYNTHETIC_DATA_DIR / "sales_daily.csv", index=False)
        inventory_snapshots.to_csv(SYNTHETIC_DATA_DIR / "inventory_snapshots.csv", index=False)
        logger.info("All 4 synthetic datasets successfully saved.")

    return sku_master, calendar, sales_daily, inventory_snapshots


if __name__ == "__main__":
    generate_all_datasets()
