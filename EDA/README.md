# Module 2 — Exploratory Data Analysis (EDA)
**Branch:** `hacao` · **Role:** Data Analyst

> This document has two parts: (A) analysis of the project requirements + where EDA sits in the pipeline, and (B) a detailed execution plan for the EDA work.

---

## PART A — REQUIREMENTS ANALYSIS

### A.1. Overall problem
Build a **regression model that predicts a property's sale price (`SalePrice`)** from structural, locational, and condition-related features, then **deploy it as a working application and monitor it after deployment** — mirroring the full lifecycle of an analytics product in industry.

Core business question:
> *"Given a property's characteristics and location, what is its estimated market price, and how confident is that estimate?"*

### A.2. Data (two required sources, combined)
1. **Kaggle — House Prices: Advanced Regression Techniques** (required base, already in `data/`):
   - `train.csv`: 1460 rows × 81 columns (79 features + `Id` + `SalePrice`).
   - `test.csv`: 1459 rows (no `SalePrice`).
   - `data_description.txt`: data dictionary explaining every column.
   - Homes in Ames, Iowa (USA).
2. **Synthetic data (required extension — generated in Python by Module 1):**
   - Neighborhood amenity scores (distance to school/hospital/transit), renovation history, days-on-market, macro indicators (interest rate, price index over time).
   - Must ship with a full **data dictionary** (column name, type, unit, valid range, generation logic).

> ⚠️ **Key point for EDA:** the final analysis dataset = **Kaggle base + Module 1's synthetic data**. See A.5 for handling this dependency.

### A.3. The 8 modules & ownership

| # | Module | Key output |
|---|--------|------------|
| 1 | Business Understanding & Data Generation | Synthetic data + data dictionary + KPIs |
| **2** | **EDA (this module)** | **Notebook + insight report + visualizations** |
| 3 | Data Cleaning | Clean dataset + cleaning decision log |
| 4 | Feature Engineering | Derived features, encoding, feature selection |
| 5 | Model Development | Models + evaluation (RMSE/MAE/MAPE/R²) |
| 6 | Deployment | API (FastAPI/Flask) + Streamlit demo + live link |
| 7 | Monitoring | Drift dashboard + retrain triggers (Evidently) |
| 8 | Final Report & Presentation | PDF report + PPTX slides + demo |

### A.4. Grading criteria (relevant to EDA)
The brief is explicit: **grading emphasizes the rigor and clarity of each stage, NOT model accuracy alone.** For Module 2, two criteria apply directly:
- ✅ **Depth of EDA and clarity of insights communicated through visualization.**
- ✅ Contributes to the "quality and realism of synthetic data" criterion — EDA is where we validate whether the synthetic data is plausible.

> 👉 Top priority: **every chart must lead to a business-meaningful insight** — no charts for the sake of charts.

### A.5. Where EDA sits in the pipeline — dependencies & handoffs

```
Module 1                    Module 2 = EDA                  Module 3               Module 4
─────────────────           ───────────────────────         ────────────────       ────────────────
Generate synthetic    ───►  Distributions, correlation, ──► Clean based on      ──► Feature engineering
data + dictionary           outliers, multicollinearity     outliers & missing      based on correlation
                                                            EDA surfaced             & skewness EDA found
```

- **Input I need (from Module 1):** the synthetic data merged with Kaggle + its data dictionary. *If not ready yet*, start EDA on the Kaggle base first, then layer in the synthetic part later (keep the notebook modular so it's easy to plug in).
- **Output I hand off (critical — Modules 3 & 4 depend on it directly):**
  - **List of columns with missing values** + rates → to Module 3.
  - **List of outliers / implausible values** (e.g. homes with huge `GrLivArea` but low price) → to Module 3.
  - **Correlation table vs `SalePrice`** + the most important feature groups → to Module 4.
  - **Multicollinear feature pairs** (e.g. `GarageCars`↔`GarageArea`, `TotalBsmtSF`↔`1stFlrSF`) → to Module 4.
  - **Confirmed skewness of `SalePrice`** → suggests log-transform for Modules 4/5.

---

## PART B — EDA EXECUTION PLAN

> ### ✅ Execution status (updated 2026-07-04)
> Steps 0–5 and 7 are **done** on the Kaggle base. Step 6 (synthetic data) is **pending Module 1**.
> Deliverables: notebook [01_eda.ipynb](01_eda.ipynb) · written insights [EDA_FINDINGS.md](EDA_FINDINGS.md) ·
> 10 charts in [figures/](figures/) · handoff files in [outputs/](outputs/) (`missing_report.csv`,
> `outlier_candidates.csv`, `correlation_top.csv`, `collinear_pairs.csv`, `eda_summary.json`).

### B.0. Proposed folder structure (inside `EDA/`)
```
EDA/
├── README.md                  ← this file (requirements + plan)
├── 01_eda.ipynb               ← main notebook (the analysis)
├── figures/                   ← exported .png charts for the Module 8 report
├── outputs/
│   ├── missing_report.csv     ← handoff to Module 3
│   ├── outlier_candidates.csv ← handoff to Module 3
│   └── correlation_top.csv    ← handoff to Module 4
└── EDA_FINDINGS.md            ← written insight summary (handoff & for slides)
```

### B.1. Specific technical requirements of Module 2 (from the brief)
1. Analyze **distributions, outliers, and correlations** between property attributes and sale price.
2. **Compare price patterns across location, property type, and age.**
3. Visualize relationships (e.g. price vs area, price vs distance-to-amenities) and **check for multicollinearity.**

### B.2. Detailed execution checklist

**Step 0 — Setup & load** *(0.5 day)* — ✅ done
- [x] Create the notebook; import `pandas, numpy, matplotlib, seaborn, scipy`.
- [x] Load `train.csv`; cross-reference `data_description.txt` to understand column meanings.
- [x] Classify columns: numeric vs categorical vs ordinal (many "quality" columns like `ExterQual`, `BsmtQual`… are **ordinal**: Ex>Gd>TA>Fa>Po — handle with care).
- [ ] (When available) merge Module 1's synthetic data.

**Step 1 — Target variable `SalePrice`** *(0.5 day)* — ✅ done
- [x] Histogram + KDE of `SalePrice` → note the right skew.
- [x] Compute skewness & kurtosis; plot `log(SalePrice)` to show the log-transform brings it close to normal.
- [x] Q-Q plot. → **Handoff insight for Modules 4/5: use a log-transform on the target.** *(skew 1.88 → 0.12)*

**Step 2 — Missing values** *(0.5 day)* — ✅ done
- [x] Table of % missing per column, sorted descending; missing bar/heatmap.
- [x] Distinguish "NA means absent" (e.g. `PoolQC`, `Alley`, `FireplaceQu` → NA = no such amenity) vs "genuinely missing" (e.g. `LotFrontage`, `GarageYrBlt`). *(19 cols missing; only 3 genuine)*
- [x] **Export `outputs/missing_report.csv` → handoff to Module 3.**

**Step 3 — Numeric variables** *(1 day)* — ✅ done
- [x] Histogram + boxplot for key numeric variables (`GrLivArea`, `TotalBsmtSF`, `LotArea`, `1stFlrSF`, `GarageArea`…).
- [x] Scatter **price vs area** (`GrLivArea` vs `SalePrice`) → catch this dataset's well-known outliers (2 homes >4000 sqft with very low prices). *(Id 524 & 1299)*
- [x] **Export `outputs/outlier_candidates.csv` → handoff to Module 3.**

**Step 4 — Correlation & multicollinearity** *(1 day)* — ✅ done
- [x] Correlation heatmap across numeric variables and vs `SalePrice`.
- [x] Top 10–15 features most correlated with `SalePrice` (expected: `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `YearBuilt`…). *(confirmed; ordinal quality cols also rank high)*
- [x] Identify multicollinear pairs (high corr between predictors). If time permits, compute preliminary VIF. *(10 pairs |r|≥0.7; VIF done)*
- [x] **Export `outputs/correlation_top.csv` + collinear-pair list → handoff to Module 4.**

**Step 5 — Categorical variables & group comparisons** *(1 day)* — ✅ done
- [x] Boxplot **`SalePrice` by `Neighborhood`** (location) → most vs least expensive areas. *(3.58× spread)*
- [x] Boxplot **`SalePrice` by `BldgType`/`HouseStyle`** (property type).
- [x] **`SalePrice` by property age**: create a temp `Age = YrSold − YearBuilt`, scatter/binned line → price vs age. *(corr −0.52)*
- [x] Boxplot by `OverallQual` (a very strong driver). *(median $50k→$432k across qual 1→10)*

**Step 6 — Analyze the synthetic data (once Module 1 is done)** *(0.5 day)*
- [ ] Check the distributions of synthetic variables (amenity score, days-on-market, interest rate…) for plausibility.
- [ ] Scatter **price vs distance-to-amenities** (as the brief suggests).
- [ ] Verify the synthetic ↔ `SalePrice` correlations point the expected business direction (closer to school/hospital → higher price?). → feed back to Module 1 if the generation logic looks unrealistic.

**Step 7 — Synthesis & handoff** *(0.5 day)* — ✅ done (PR optional)
- [x] Write `EDA_FINDINGS.md`: 8–12 key insights, each with a one-line business meaning + chart link. *(10 insights)*
- [x] Export all key charts to `figures/` (reused by Module 8 for slides/report). *(10 charts)*
- [x] Finalize the files in `outputs/` for Modules 3 & 4. *(+ `collinear_pairs.csv`, `eda_summary.json`)*
- [x] Commit & push branch `hacao`. *(open a PR when the team is ready)*

### B.3. Time estimate
Total ~**5–6 working days**. Start immediately on the Kaggle base (Steps 0–5); insert Step 6 once Module 1 delivers the synthetic data.

### B.4. Quality principles (to hit the "depth & clarity" criterion)
1. **Every chart = one insight.** No bare charts without commentary.
2. **Prioritize business meaning** over raw technicality: "NridgHt is ~3× more expensive than MeadowV" > "corr = 0.79".
3. **Consistent chart style** (same palette, titles/labels/units) so they can be reused in the final report.
4. **Every actionable finding must be written to a handoff file** — EDA isn't just for me; it's the input for Module 3 (cleaning) and Module 4 (feature engineering).

### B.5. Dependencies to coordinate proactively
- **From Module 1:** when will the synthetic data + dictionary be ready? What is the schema of the synthetic columns?
- **With Module 3:** agree on the format of `missing_report.csv` / `outlier_candidates.csv` so it can be used directly.
- **With Module 4:** agree on the correlation format they need (top-N vs full matrix).
