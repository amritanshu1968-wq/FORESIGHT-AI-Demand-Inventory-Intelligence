# PROJECT FORESIGHT — EXECUTIVE READOUT & STRATEGIC MEMO

> **IMPORTANT DATA PROVENANCE & DEMONSTRATION DISCLOSURE**  
> **Data Source:** Synthetic retail dataset generated for project development and demonstration. All forecasts, risk scores, financial impacts, and operational insights presented in this memo are demonstration outputs and do not represent actual commercial figures of NorthBay Living or Zidio Development.

---

## 1. Executive Summary

Project FORESIGHT delivers an AI-Powered Demand & Inventory Intelligence Platform designed to solve the chronic retail dilemma of balancing inventory availability against working capital liquidity. Developed around the retail portfolio scenario of "NorthBay Living" (300 SKUs across Small Appliances, Décor & Lighting, Home Furnishings, and Kitchen & Dining), FORESIGHT replaces static heuristics with machine learning forecasting and a multi-factor 2D Inventory Risk Engine.

### Key Headline Results:
- **Forecast Superiority:** Machine learning (`HistGradientBoostingRegressor`) achieves a **38.22% WAPE**, outperforming the Seasonal Naive baseline (68.78% WAPE) by **+44.43% relative accuracy gain** across 9,600 out-of-sample backtest evaluations.
- **Risk Segmentation:** The 2D Decision Grid identifies **87 SKUs (29.0%) in immediate need of replenishment** and **18 SKUs (6.0%) holding surplus stock**.
- **Working Capital & Margin Opportunity:** Identifies **₹6.66M in trapped excess working capital** across overstocked SKUs and protects **₹2.12M in gross margin at imminent risk of loss** due to stockouts.
- **Quality & Verification:** The entire platform is backed by **32 automated tests (100% pass rate)**, with local interactive dashboarding (Streamlit) and API scoring (FastAPI) verified.

---

## 2. The Strategic Inventory Challenge

In modern retail, static 30-day moving averages and uncalibrated safety stock buffers lead to systematic failures:
1. **Stockouts on High-Demand SKUs:** Fast-moving products stock out during promotional campaigns (+38.37% lift) and festive spikes (+40.90% surge). Over 3 years of transactional history, unmet demand occurred on **27.63% of daily SKU observations** (90,846 SKU-days), with Small Appliances recording a 37.58% stockout rate.
2. **Capital Lockup in Slow Movers:** Excess inventory exceeding 10 weeks of cover accumulates silently, incurring 18–25% annual carrying costs and necessitating steep markdowns.
3. **Lead-Time Exposure:** Catalog supplier lead times range from 10 to 45 days. Reordering without forward-looking demand projections guarantees severe stockouts during replenishment lead times.

---

## 3. Demand Forecasting Benchmark

Models were benchmarked using a 4-fold rolling-origin walk-forward protocol across 9,600 out-of-sample test instances:

| Model Architecture | WAPE (%) | MAE (Units) | RMSE (Units) | Bias (%) | Relative WAPE Gain vs Baseline |
|---|---|---|---|---|---|
| **HistGradientBoostingRegressor** | **38.22%** | **10.43** | **18.72** | **+0.85%** | **+44.43% Improvement** |
| Ridge Regression (L2) | 53.06% | 14.49 | 24.88 | +2.34% | +22.86% Improvement |
| 4-Week Moving Average | 62.38% | 17.03 | 28.78 | +2.03% | +9.30% Improvement |
| Seasonal Naive (52w) | 68.78% | 18.78 | 35.11 | -1.01% | Benchmark Baseline (0.0%) |
| Last-Week Naive | 79.45% | 21.69 | 40.39 | +2.03% | -15.51% (Inferior) |

**Operational Takeaway:** Tree-based gradient boosting captures non-linear promotional, seasonal, and trend signals while maintaining near-zero bias (+0.85%), preventing chronic under-replenishment.

---

## 4. The 2D Inventory Risk Matrix & Portfolio Segmentation

The FORESIGHT 2D Risk Engine pairs forward demand projections with a **tiered safety stock model**:
- **0.95 Target Service Level ($Z = 1.6449$):** 186 standard SKUs (62.0% of catalog)
- **0.98 Target Service Level ($Z = 2.0537$):** 58 critical/high-margin Class A SKUs (19.3% of catalog)
- **0.90 Target Service Level ($Z = 1.2816$):** 56 slow-moving SKUs (18.7% of catalog)

$$\text{Safety Stock} = \max\left(1, \left\lceil Z \times \sigma_{\text{err}} \times \sqrt{L_{\text{weeks}}} \right\rceil\right)$$

### Decision Grid Distribution (300 SKUs):
1. **Reorder Now (87 SKUs / 29.0%):** Immediate stockout risk ($S \ge 0.50$). Requires **12,351 replenishment units** to restore safety buffers.
2. **Watch / Volatile (36 SKUs / 12.0%):** High demand volatility ($CV \ge 0.85$). Requires supplier lead-time audits before committing capital.
3. **Healthy (159 SKUs / 53.0%):** Balanced inventory ($4 \le WOC \le 10$). Standard automated replenishment cadence.
4. **Markdown / Clear (18 SKUs / 6.0%):** Surplus stock ($WOC > 10$ weeks). Prescribes targeted clearance on **1,545 excess units**.

---

## 5. Illustrative Financial Exposure from Synthetic Planning Environment

| Product Category | Catalog SKUs | Current Inventory Value (₹) | Lost Margin Risk (₹) | Excess Capital Locked (₹) |
|---|---|---|---|---|
| **Small Appliances** | 75 SKUs | ₹17,016,696.11 | ₹1,475,825.12 (69.5%) | ₹2,933,994.86 (44.0%) |
| **Décor & Lighting** | 75 SKUs | ₹8,973,022.81 | ₹309,296.90 (14.6%) | ₹1,001,835.00 (15.0%) |
| **Home Furnishings** | 75 SKUs | ₹9,244,777.03 | ₹242,434.53 (11.4%) | ₹1,105,590.94 (16.6%) |
| **Kitchen & Dining** | 75 SKUs | ₹7,549,417.52 | ₹97,164.57 (4.6%) | ₹1,622,214.41 (24.3%) |
| **Portfolio Total** | **300 SKUs** | **₹42,783,913.47** | **₹2,124,721.12** | **₹6,663,635.21** |

---

## 6. Implementation & Operational Delivery

- **Interactive Decision Cockpit:** Streamlit planning dashboard (`dashboard/app.py`) featuring Executive KPIs, Demand Forecast Visualizer with 80% empirical intervals, 2D Risk Matrix, Priority Action Queues, and Data Quality Profiling.
- **REST Inference API:** FastAPI service (`api/main.py`) with Pydantic v2 schemas and Swagger UI documentation (`/docs`). Models and artifacts are loaded into memory during startup to reduce repeated file/model loading during requests.
- **Deployment Status:** Both services are currently verified in the local host environment (`http://localhost:8501` and `http://127.0.0.1:8000`). Public cloud deployment is pending and will be completed as Zidio Deliverable #2.
- **Quality Assurance:** Backed by 32 passing automated unit tests in `pytest tests/` (100% pass rate).

---

## 7. Next Steps & Commercial Roadmap

1. **Deploy Public Showcase:** Host the interactive dashboard and scoring API on public cloud infrastructure (Deliverable #2).
2. **Enterprise Connectors:** Integrate pre-built adapters for enterprise ERP/WMS platforms (SAP, NetSuite, Shopify).
3. **Automated Purchase Orders:** Connect API scoring outputs directly to procurement purchase order workflows.
4. **Continuous MLOps:** Deploy automated data drift detection and model retraining pipelines.
