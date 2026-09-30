# Forecast Model Evaluation & Backtest Report: Project FORESIGHT

> **Document Status**: Complete / Backtested  
> **Prepared For**: Head of Operations & Finance Lead  
> **Prepared By**: Data Science & Analytics Team  
> **Evaluation Framework**: 4-Fold Rolling-Origin Walk-Forward Cross-Validation  

---

## 1. Forecast Objective & Business Horizon

* **Grain**: Weekly SKU-level customer demand (`sku_id` + `week_start`).
* **Forward Horizon**: 12 Weeks (Quarterly operational planning and supplier reorder cycle).
* **Primary Metric**: **WAPE** (Weighted Absolute Percentage Error) — robust to low-volume and intermittent SKUs where MAPE explodes.
* **Secondary Metrics**: MAE, RMSE, Signed Bias (tracking systematic over- or under-stocking tendencies).
* **Benchmark Hurdle**: The production ML model must beat a **Seasonal Naive (52-week lag)** baseline under strict out-of-sample backtesting.

> [!IMPORTANT]
> **Data Provenance Disclosure**:
> Data Source: Synthetic retail dataset generated for project development and demonstration. These findings demonstrate the capabilities of the FORESIGHT platform and should not be interpreted as actual NorthBay Living or Zidio Development business results.

---

## 2. Walk-Forward Backtesting Architecture

To prevent temporal data leakage, standard random train/test splits were strictly prohibited. Instead, the models were evaluated using **Rolling-Origin Cross-Validation** across 4 sequential operational origins with 8-week holdout windows:

```
Full History (158 Weeks)
Fold 1: [Train: Weeks 1..126 (126w)] -> [Test: Weeks 127..134 (8w)]
Fold 2: [Train: Weeks 1..134 (134w)] -> [Test: Weeks 135..142 (8w)]
Fold 3: [Train: Weeks 1..142 (142w)] -> [Test: Weeks 143..150 (8w)]
Fold 4: [Train: Weeks 1..150 (150w)] -> [Test: Weeks 151..158 (8w)]
```

* **Evaluated Records**: **9,600 out-of-sample forecast instances** (300 SKUs x 8 weeks x 4 folds).
* **Zero Leakage Enforcement**: SKU scale normalization parameters and model weights were re-fitted independently at each origin using only history strictly prior to or at the origin date.

---

## 3. Overall Model Comparison & Measured Results

| Forecasting Model | WAPE (%) | MAE (Units) | RMSE (Units) | Signed Bias (%) |
| :--- | :--- | :--- | :--- | :--- |
| **HistGradientBoosting** | **38.22%** | 10.43 | 18.72 | +0.85% |
| **Ridge Regression** | **53.06%** | 14.49 | 24.88 | +2.34% |
| **Moving Average (4w)** | **62.38%** | 17.03 | 28.78 | +2.03% |
| **Seasonal Naive (52w)** | **68.78%** | 18.78 | 35.11 | -1.01% |
| **Last-Week Naive** | **79.45%** | 21.69 | 40.39 | +2.03% |

### Key Performance Findings:
1. **Clear Baseline Victory**: The **HistGradientBoosting** achieved a **WAPE of 38.22%**, outperforming the Seasonal Naive baseline (68.78%) by **44.43% relative accuracy gain**.
2. **Reduced Variance & Error**: The ML forecaster achieved substantial reductions in MAE and RMSE compared to moving averages and last-week naive methods, successfully filtering stochastic weekly demand noise.
3. **Controlled Forecast Bias**: Signed Bias remains strictly within +/- 2.5%, confirming that the model does not exhibit systematic over-forecasting (which causes overstock) or under-forecasting (which triggers stockouts).

---

## 4. Category-Level Performance Breakdown

| Product Category | Seasonal Naive WAPE | HistGBM (ML) WAPE | Relative Accuracy Gain |
| :--- | :--- | :--- | :--- |
| **Décor & Lighting** | 70.73% | 38.66% | **+45.34%** |
| **Home Furnishings** | 75.08% | 40.34% | **+46.27%** |
| **Kitchen & Dining** | 54.71% | 32.22% | **+41.11%** |
| **Small Appliances** | 77.5% | 43.31% | **+44.12%** |

* **Category Observations**: The machine learning model captures seasonal turning points and festival surges significantly better than static 52-week repeats, especially in **Small Appliances** and **Décor & Lighting** where promotional discounts and holiday campaigns drive non-linear demand lifts.

---

## 5. Leakage Prevention & Audit Verification

1. **Lag Enforcement**: Feature definitions enforce t - k (k >= 1). Current week actuals never enter the feature set.
2. **Rolling Statistics**: Rolling mean and std features use shift(1), guaranteeing that the 4-week and 13-week moving windows only look backward from t-1.
3. **Scale Normalization Isolation**: Historical mean demand scales are calculated strictly within the training window t <= T_origin. Future test volumes never influence scale factors.
4. **Independent Origin Training**: Tree structures and linear weights are retrained from scratch at each fold.

---

## 6. Forward Multi-Step Forecasting Methodology

* **Production Model**: `HistGradientBoostingRegressor` trained on all 158 weeks of historical data.
* **Forward Output**: **3,600 forward forecasts** generated (300 active SKUs x 12 forward weeks) saved to [`outputs/forecasts/forecast_results.csv`](file:///c:/Users/Amritanshu/OneDrive/Desktop/zidio%20project%201/outputs/forecasts/forecast_results.csv).
* **Uncertainty Intervals**: Calibrated 80% empirical prediction intervals ([y_hat - 1.282 * sigma_resid, y_hat + 1.282 * sigma_resid]) provide operations with best-case and worst-case demand bounds for safety stock sizing.

---

## 7. Operational Limitations & Next Steps

1. **Intermittent SKU Sparsity**: For sporadic Class C SKUs with zero sales in >80% of weeks, point forecasts naturally compress toward fractional averages (e.g. 0.3 units/week). Croston's method or hurdle models can be explored in future iterations.
2. **Marketing Pipeline Coupling**: The forward forecast assumes scheduled festive campaign participation. Actual promotions must be confirmed by merchandising during monthly S&OP meetings.

---
*Report generated automatically by Project FORESIGHT Forecasting & Backtesting Engine.*
