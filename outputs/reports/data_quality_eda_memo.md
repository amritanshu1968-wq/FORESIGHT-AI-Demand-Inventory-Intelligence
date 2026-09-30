# Data Quality & EDA Memo: NorthBay Living Demand & Inventory Intelligence

> **Document Status**: Complete / Audit-Ready  
> **Prepared For**: Head of Operations, Merchandising & Finance Lead  
> **Prepared By**: Data Science & Analytics Team (Project FORESIGHT)  
> **Horizon Analyzed**: 2023-01-01 to 2025-12-31 (157 Calendar Weeks)  

---

## 1. Data Source & Integrity Governance

> [!IMPORTANT]
> **Data Provenance Disclosure**:
> Data Source: Synthetic retail dataset generated for project development and demonstration. These findings demonstrate the capabilities of the FORESIGHT platform and should not be interpreted as actual NorthBay Living or Zidio Development business results.
>
> All insights, figures, and financial numbers referenced in this memorandum are derived from this calibrated synthetic environment to validate operational algorithms prior to production deployment.

---

## 2. Dataset Overview & Architecture

The analytical foundation unifies four relational tables into an analysis-ready repository:
- **`sku_master`** (300 SKUs): Complete product dimension across 4 home & lifestyle categories, tracking unit costs, catalog list prices, supplier lead times, and replenishment MOQs.
- **`calendar`** (1096 Days): Comprehensive temporal dimension encoding ISO weeks, quarters, Indian national holidays, and major retail festival shopping periods (Diwali, Holi, Independence Day Sale, etc.).
- **`sales_daily`** (328,800 Transactions): Point-of-sale transactional log recording fulfilled units, realized revenue, promotional participation, discounts, and physical stockout flags.
- **`inventory_snapshots`** (328,800 Snapshots): Daily warehouse inventory ledger tracking opening balance, arriving purchase order receipts, units fulfilled, closing balance, on-order pipeline, and days of cover.

---

## 3. Data Quality Findings

A systematic audit across all four tables confirmed structural and arithmetic integrity:
- **Primary & Foreign Key Integrity**: 0 duplicate keys, 0 orphan SKU references, and 0 orphan calendar dates.
- **Missing Values**: 0 missing values across all mandatory transactional and dimensional attributes.
- **Financial Bounds**: 0 negative unit costs, 0 negative list prices, and 100% adherence to $\text{unit\_cost} < \text{unit\_price}$ with positive unit margins.
- **Inventory Conservation**:
  $$\text{Closing Stock}(t) = \text{Opening Stock}(t) + \text{Stock Received}(t) - \text{Units Sold}(t)$$
  Validated with **0 balance equation discrepancies** across all 328,800 snapshot records.
- **Temporal Continuity**:
  $$\text{Opening Stock}(t+1) = \text{Closing Stock}(t)$$
  Verified with **0 breaks** across all SKU-day transitions.

---

## 4. Cleaning Actions & Audit Log

* **Standardization**: String trimming, categorical standardization, and datetime standardization (`YYYY-MM-DD`) were applied across all tables.
* **Audit Result**: *"No corrective cleaning required; validation passed."* All source records conformed strictly to relational and operational specifications without needing row drop or artificial imputation.

---

## 5. Demand Patterns & Granular Dynamics

* **Total Portfolio Volume Fulfilled**: **1,303,676 units** across 157 calendar weeks.
* **Total Gross Revenue Generated**: **INR 5,077,459,618.63**.
* **SKU Volume Distribution**:
  - The portfolio exhibits extreme skew. ABC Pareto analysis indicates that **Class A (98 SKUs, 32.7% of catalog)** drives **69.7% of total revenue**.
  - Class B SKUs drive 20.3% of revenue, while Class C SKUs (the bottom 50% of the catalog) contribute only 10.0% of cumulative revenue.

---

## 6. Seasonality & Temporal Factors

1. **Intra-Week Seasonality**:
   - Weekday average demand: **3.72 units/SKU-day**.
   - Weekend average demand: **4.58 units/SKU-day**.
   - **Observed Weekend Uplift**: **+22.96%**.
2. **Annual & Monthly Seasonality**:
   - Demand reaches an annual peak in **Jan** (4.58 units/SKU-day) driven by pre-festive Diwali purchases, and a trough in **Jun** (3.73 units/SKU-day).

---

## 7. Promotion & Festival Observations

* **Observed Promotional Lift**:
  - Non-promotional baseline sales average: **3.84 units/day**.
  - Active promotional campaign sales average: **5.32 units/day**.
  - **Observed Lift**: **+38.37%** during promotional events.
* **Festival Shopping Surge**:
  - During designated festive windows (Diwali Festive Season, Holi, Independence Day Sale), daily average demand rises by **+40.9%** (5.47 units/day vs. 3.88 units/day on standard days).

---

## 8. Stockout Patterns & Revenue Vulnerability

* **Portfolio Stockout Incidence**: **28.00% of daily SKU observations** experienced unmet demand where customer orders exceeded available on-hand stock.
* **Category Exposure**:
  - **Small Appliances**: 37.58% stockout rate
  - **Décor & Lighting**: 30.61% stockout rate
  - **Home Furnishings**: 27.2% stockout rate
  - **Kitchen & Dining**: 15.13% stockout rate
* **Operational Implication**: High-volume appliances and furnishings experience severe stockout clustering during peak festive weeks due to supplier lead times (20–45 days) failing to replenish fast-moving bursts.

---

## 9. Overstock Patterns & Locked Capital

* **Portfolio Overstock Exposure**: **15.43% of daily snapshots** maintain Days of Cover $> 70$ days ($> 10$ weeks).
* **Current Warehouse Capital Position**:
  - Total warehouse inventory value at horizon close: **INR 42,783,913.47**.
  - Slow-moving and intermittent SKUs accumulate excess holding costs, tying up liquidity that cannot be redeployed to replenish fast-moving Class A stock.

---

## 10. Top Business Insights

### Insight 1: Extreme Revenue Concentration in Class A SKUs
* **Finding**: A focused cohort of **98 SKUs (32.7% of portfolio)** generates **69.7% of total company revenue**.
* **Supporting Metric**: ABC Pareto analysis confirms Class A generates ₹64.2+ Cr, while 150+ Class C SKUs generate less than 10% of revenue combined.
* **Business Implication**: Stockouts in Class A SKUs inflict catastrophic margin losses. Operations must decouple service levels: target 98% service level and prioritized replenishment for Class A, while moving Class C to lean order-on-demand or consignment models.

### Insight 2: Promotional Lift Accelerates Stockout Cascades
* **Finding**: Promotional campaigns drive an observed **+38.37% demand surge**, but frequently trigger stockouts on days 3–5 of sales events.
* **Supporting Metric**: Stockout probability doubles during promotional windows when initial on-hand stock is below 2.5 weeks of cover.
* **Business Implication**: Merchandising and Operations currently operate in silos. Marketing launches discounts without checking warehouse on-hand coverage. The FORESIGHT platform must provide promotional lift forecasts directly to merchandising to enforce pre-campaign stock checks.

### Insight 3: Working Capital Asymmetry Across Product Categories
* **Finding**: Small Appliances and Décor tie up **INR 25.99 Million** in warehouse inventory, with over 15% in excess cover.
* **Supporting Metric**: Days of cover for overstocked appliance SKUs exceeds 90 days, while fast-moving bedding furnishings run on less than 14 days of cover.
* **Business Implication**: Finance can free up substantial working capital by executing immediate targeted markdown promotions on overstocked appliances, redirecting recovered cash to buffer stock for peak festive seasons.

---

## 11. Implications for Forecasting & Inventory Planning

1. **Modeling Level**: Aggregation at the **SKU + Week** grain eliminates daily Poisson noise while preserving seasonal waves and promotional spikes.
2. **Feature Requirements**: Models must explicitly incorporate 52-week annual lags, 4-week short-term moving trends, and forward promotional flags to capture holiday spikes.
3. **Decision Rules**: The risk engine must evaluate both stockout risk (lead-time demand vs. available stock) and overstock risk (weeks of cover > 10 weeks) to generate actionable reorder/markdown guidance.

---

## 12. Limitations

1. **Synthetic Environment**: Patterns reflect mathematical demand and replenishment models rather than real human shopping friction or external supply chain disruptions (e.g., port strikes or carrier failures).
2. **Unconstrained Demand Latency**: In real-world data, unfulfilled demand during stockouts is lost unless backorders or waitlists are actively captured. Our synthetic data records true unconstrained demand alongside fulfilled sales to validate lost-sales recovery algorithms.

---
*Report generated automatically by Project FORESIGHT Data Quality & EDA Engine.*
