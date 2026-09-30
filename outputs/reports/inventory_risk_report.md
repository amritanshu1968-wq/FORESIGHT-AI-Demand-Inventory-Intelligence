# Inventory Risk & Capital Optimization Report: Project FORESIGHT

> **Document Status**: Complete / Executive Ready  
> **Prepared For**: Head of Operations & Finance Lead  
> **Prepared By**: Data Science & Analytics Team  
> **Effective Horizon**: 12-Week Forward Operational Window  

---

## 1. Executive Summary & Headline Financial Impact

Through integration of ML weekly demand forecasts with warehouse stock ledgers, the **FORESIGHT Risk Engine** evaluated all 300 SKUs against dynamic safety stock thresholds, weeks of cover, and supplier lead times:

* **Total Current Warehouse Inventory Value**: **INR 42,783,913.47**
* **Total Gross Margin at Risk (Stockout Threat)**: **INR 2,124,721.12**
* **Total Working Capital Locked in Excess Stock**: **INR 6,663,635.21**
* **Total Capital Optimization Opportunity**: **INR 8,788,356.33**

> [!IMPORTANT]
> **Data Provenance Disclosure**:
> Data Source: Synthetic retail dataset generated for project development and demonstration. These findings demonstrate the capabilities of the FORESIGHT platform and should not be interpreted as actual NorthBay Living or Zidio Development business results.

---

## 2. 2D Decision Grid Distribution

Every SKU is assigned to exactly one operational quadrant based on its normalized stockout score ($Y$-axis) and overstock score ($X$-axis):

| Decision Quadrant | Operational Meaning | SKU Count | Share of Portfolio | Primary Action |
| :--- | :--- | :--- | :--- | :--- |
| **Reorder Now** | High Stockout, Low Overstock | **87 SKUs** | **29.0%** | Urgent purchase order to prevent stockout |
| **Markdown / Clear** | Low Stockout, High Overstock | **18 SKUs** | **6.0%** | Targeted promotion to liquidate locked capital |
| **Watch / Volatile** | High on Both / Erratic Demand | **36 SKUs** | **12.0%** | Manual audit; investigate supplier jitter |
| **Healthy** | Low Stockout, Low Overstock | **159 SKUs** | **53.0%** | Routine monitoring; standard replenishment |

---

## 3. Category-Level Risk & Financial Exposure

| Product Category | Total SKUs | Reorder Candidates | Markdown Candidates | Total Inventory Value | Lost Margin at Risk | Locked Excess Capital |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Décor & Lighting** | 75 | 22 | 2 | INR 8,973,022.81 | INR 309,296.90 | INR 1,001,835.00 |
| **Home Furnishings** | 75 | 20 | 5 | INR 9,244,777.03 | INR 242,434.53 | INR 1,105,590.94 |
| **Kitchen & Dining** | 75 | 11 | 7 | INR 7,549,417.52 | INR 97,164.57 | INR 1,622,214.41 |
| **Small Appliances** | 75 | 34 | 4 | INR 17,016,696.11 | INR 1,475,825.12 | INR 2,933,994.86 |

---

## 4. Priority Operational Action Lists

### A. Top 5 Urgent Reorder Candidates (High Lost-Margin Risk)
| SKU ID | Product Name | Category | On-Hand | ROP | Rec. Order Qty | Lost Margin at Risk (INR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SKU-0272** | NorthBay Studio Cold Press Juicers #272 | Small Appliances | 0 | 255 | **242** | INR 196,840.41 |
| **SKU-0280** | NorthBay Basics Electric Kettles #280 | Small Appliances | 0 | 622 | **281** | INR 172,782.71 |
| **SKU-0284** | NorthBay Reserve Air Fryers #284 | Small Appliances | 0 | 152 | **112** | INR 122,181.75 |
| **SKU-0192** | NorthBay Basics Espresso Machines #192 | Small Appliances | 155 | 344 | **273** | INR 115,405.15 |
| **SKU-0300** | NorthBay Studio Espresso Machines #300 | Small Appliances | 0 | 201 | **201** | INR 107,713.29 |

### B. Top 5 Markdown / Clearance Candidates (High Excess Working Capital)
| SKU ID | Product Name | Category | On-Hand | Weeks of Cover | Excess Units | Capital Locked (INR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SKU-0114** | NorthBay Basics Table Lamps #114 | Décor & Lighting | 168 | 31.1 wks | **133** | INR 467,518.94 |
| **SKU-0160** | NorthBay Reserve Espresso Machines #160 | Small Appliances | 418 | 10.6 wks | **47** | INR 420,201.62 |
| **SKU-0251** | NorthBay Basics Cast Iron Cookware #251 | Kitchen & Dining | 197 | 29.1 wks | **147** | INR 285,504.87 |
| **SKU-0127** | NorthBay Studio Dinnerware Sets #127 | Kitchen & Dining | 793 | 10.1 wks | **167** | INR 220,758.97 |
| **SKU-0260** | NorthBay Studio Cold Press Juicers #260 | Small Appliances | 107 | 14.5 wks | **43** | INR 218,876.88 |

---

## 5. Mathematical & Policy Methodology

1. **Safety Stock Formulation**:
   $$SS = \lceil Z \cdot \sigma_{\text{weekly\_error}} \cdot \sqrt{\text{Lead Time (Weeks)}} \rceil$$
   Calibrated using SKU-specific forecast residual variance from backtest evaluation and target service levels ($Z \in [1.28, 2.05]$).
2. **Reorder Point (ROP)**:
   $$\text{ROP} = \lceil \hat{d}_{\text{weekly}} \cdot \text{Lead Time (Weeks)} + SS \rceil$$
   Driven strictly by forward model predictions, eliminating reliance on lagging historical averages.
3. **Lost Margin Risk Calculation**:
   $$\text{Lost Margin Risk (INR)} = \max(0, \text{LTD} - \text{Effective Available}) \times (\text{Price} - \text{Cost})$$
   Reflects true bottom-line profitability lost rather than gross revenue.
4. **Excess Capital Locked Calculation**:
   $$\text{Excess Capital Locked (INR)} = \max(0, \text{On-Hand} - \text{Target Stock}) \times \text{Cost}$$

---

## 6. Assumptions & Limitations

1. **Ordering Cost Constant**: Assumed standard procurement administration cost $S = \text{INR } 500$ per PO line item for EOQ sizing.
2. **Holding Cost Rate**: Standard 20% annual cost of capital and warehousing applied to unit cost.
3. **Supplier Lead Time Rigidity**: Lead times are treated as deterministic; real supply chains experience shipping variance requiring continuous buffer recalibration.

---
*Report generated automatically by Project FORESIGHT Inventory Intelligence Engine.*
