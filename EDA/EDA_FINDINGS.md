# EDA Findings — House Prices (Ames, Iowa)

**Module 2 · Branch `hacao` · Data Analyst**
Source: `data/kaggle_data/train.csv` (1460 rows × 81 cols) for Steps 0–5,7 · `data/final_data/train_v2.csv`
(+5 supplementary cols) for Step 6 · Notebook: [01_eda.ipynb](01_eda.ipynb)

> Written insight summary for **Module 8** (report/slides) and a **handoff** to **Module 3** (cleaning) and
> **Module 4** (feature engineering). Every insight below links to a chart in [figures/](figures/) and, where
> relevant, to a handoff file in [outputs/](outputs/). All numbers are computed by the notebook — see
> [outputs/eda_summary.json](outputs/eda_summary.json).

---

## TL;DR — the 5 numbers that matter

| Metric | Value | Implication |
|--------|-------|-------------|
| `SalePrice` skew (raw → log) | **1.88 → 0.12** | Model on **`log(SalePrice)`** |
| Columns with missing data | **19**, of which only **3** are genuine | Most `NA` = "feature absent", not a data error |
| Strongest driver | **`OverallQual`** (r = **0.79**) | Quality dominates price |
| Cheapest → priciest neighborhood (median) | **3.58×** ($88k → $315k) | Location is a must-keep feature |
| Outlier candidates (IQR scan) | **45 rows** (2 leverage + 43 multivariate) | Remove the 2 leverage; review the rest |

---

## Key insights

### 1. The target is strongly right-skewed → use a log transform
`SalePrice` ranges **\$34,900 – \$755,000** (median **\$163,000**, mean **\$180,921** — mean > median confirms the
right tail). Skewness = **1.88**, kurtosis = **6.54**. Taking `log(1+SalePrice)` drops skew to **0.12** (near-normal).
→ **Handoff to Modules 4/5: train on `log(SalePrice)`**, then exponentiate predictions.
📊 [figures/01_saleprice_distribution.png](figures/01_saleprice_distribution.png)

### 2. Missing data is mostly "structural absence", not corruption
19 columns contain `NA`, but only **3 are genuinely missing** and need imputation:
- `LotFrontage` — **17.74%** (impute, e.g. by neighborhood median)
- `MasVnrArea` — **0.55%** (impute 0 where no veneer)
- `Electrical` — **0.07%** (1 row; impute mode `SBrkr`)

The other 16 (`PoolQC` 99.5%, `MiscFeature` 96.3%, `Alley` 93.8%, `Fence` 80.8%, `FireplaceQu` 47.3%, all
`Garage*` 5.6%, all `Bsmt*` ~2.5%) encode **"the feature does not exist"** → fill with `"None"` / `0`, **do not
drop the rows**.
→ **Handoff to Module 3:** [outputs/missing_report.csv](outputs/missing_report.csv) (with the `na_meaning` column).
📊 [figures/02_missing_values.png](figures/02_missing_values.png)

### 3. Outliers: 2 leverage points to remove, + 45 candidates to review
A **systematic scan** (IQR 1.5× as primary, robust to skew; z-score `|z|>3` as reference) across 12 continuous
features, counting how many features flag each row:
- **High-leverage → remove:** `Id 524` (GrLivArea **4676** sqft, **\$184,750**) and `Id 1299` (GrLivArea
  **5642** sqft, **\$160,000**) — the two largest homes, both `OverallQual 10` in **Edwards**, both **`Partial`**
  sales. They sit far below trend and distort any linear fit.
- **Multivariate candidates → review (don't blindly drop):** 43 more rows are IQR-outliers on **≥ 3** features
  (of 1460, only 45 rows total cross that bar). Most are **genuine luxury homes** (NoRidge / StoneBr / NridgHt,
  \$500k–\$755k) — valid data, not errors.
- **Most outlier-prone features:** `MasVnrArea` (96), `LotFrontage` (88), `OpenPorchSF` (77), `LotArea` (69) —
  all heavy right tails.

→ **Handoff to Module 3:** 45 rows with `n_outlier_flags`, `flagged_features`, `reason` in
[outputs/outlier_candidates.csv](outputs/outlier_candidates.csv).
📊 [figures/03_grlivarea_vs_saleprice.png](figures/03_grlivarea_vs_saleprice.png) · [figures/03_outlier_scan.png](figures/03_outlier_scan.png)

### 4. `OverallQual` is the single strongest driver
Correlation with `SalePrice` = **0.79**. Median price climbs steeply and **non-linearly** with quality:
qual 5 → \$133k, qual 8 → \$270k, qual 10 → \$432k (qual 1 is only \$50k).
→ **Handoff to Module 4:** must-keep feature; the non-linearity favours tree models or a polynomial/spline term.
📊 [figures/05_saleprice_by_overallqual.png](figures/05_saleprice_by_overallqual.png)

### 5. Size and quality groups lead the correlation ranking
Top drivers of `SalePrice` (Pearson, numeric + encoded-ordinal):

| Rank | Feature | r | Group |
|------|---------|-----|-------|
| 1 | `OverallQual` | 0.79 | Quality |
| 2 | `GrLivArea` | 0.71 | Size |
| 3 | `ExterQual` | 0.68 | Quality |
| 4 | `KitchenQual` | 0.66 | Quality |
| 5 | `GarageCars` | 0.64 | Garage |
| 6 | `GarageArea` | 0.62 | Garage |
| 7 | `TotalBsmtSF` | 0.61 | Basement |
| 8 | `1stFlrSF` | 0.61 | Size |
| 9 | `BsmtQual` | 0.59 | Quality |
| 10 | `FullBath` | 0.56 | Rooms |

**Encoding the ordinal quality columns paid off** — `ExterQual`, `KitchenQual`, `BsmtQual` land in the top 10.
→ **Handoff to Module 4:** full ranking in [outputs/correlation_top.csv](outputs/correlation_top.csv).
📊 [figures/04_top_correlations.png](figures/04_top_correlations.png) · [figures/04_correlation_heatmap.png](figures/04_correlation_heatmap.png)

### 6. Strong multicollinearity — collapse redundant pairs before modelling
10 predictor pairs have |r| ≥ 0.70. The ones to act on:

| Pair | \|r\| | Suggested action |
|------|------|------------------|
| `GarageQual` ↔ `GarageCond` | 0.96 | Keep one |
| `PoolArea` ↔ `PoolQC` | 0.94 | Collapse to a `HasPool` flag (both ~0 everywhere) |
| `GarageCars` ↔ `GarageArea` | 0.88 | Keep one (Cars is slightly stronger) |
| `Fireplaces` ↔ `FireplaceQu` | 0.86 | Keep one |
| `YearBuilt` ↔ `GarageYrBlt` | 0.83 | Keep `YearBuilt` |
| `GrLivArea` ↔ `TotRmsAbvGrd` | 0.83 | Keep `GrLivArea` |
| `TotalBsmtSF` ↔ `1stFlrSF` | 0.82 | Keep one or sum them |

→ **Handoff to Module 4:** full list in [outputs/collinear_pairs.csv](outputs/collinear_pairs.csv).

### 7. VIF confirms a definitional trap in the area features
Preliminary VIF: `GrLivArea` = **126**, `2ndFlrSF` = **87**, `1stFlrSF` = **70**. This is because
**`GrLivArea` ≈ `1stFlrSF` + `2ndFlrSF`** (+ low-qual finished). → **Do not feed all three to a linear model
together**; keep `GrLivArea` alone, or the two floor areas without the total. Remaining features are healthy
(VIF < 6).

### 8. Location drives a 3.6× price spread
Median `SalePrice` by neighborhood ranges **3.58×**:
- **Priciest:** NridgHt \$315k · NoRidge \$302k · StoneBr \$278k · Timber \$228k · Somerst \$226k
- **Cheapest:** MeadowV \$88k · IDOTRR \$103k · BrDale \$106k · OldTown \$119k · Edwards \$122k

→ **Handoff to Module 4:** `Neighborhood` is a must-keep categorical; consider target/ordinal encoding by median price.
📊 [figures/05_saleprice_by_neighborhood.png](figures/05_saleprice_by_neighborhood.png)

### 9. Newer homes sell for more
`Age = YrSold − YearBuilt` correlates **−0.52** with `SalePrice`; median price falls steadily across age buckets.
→ **Handoff to Module 4:** derive an `Age` (and `YearsSinceRemod`) feature instead of using raw years.
📊 [figures/05_saleprice_vs_age.png](figures/05_saleprice_vs_age.png)

### 10. Property type matters less than quality/location but is not flat
`BldgType` / `HouseStyle` show single-family and 2-story homes commanding higher medians than duplexes/split
levels — a secondary effect worth one-hot encoding.
📊 [figures/05_saleprice_by_property_type.png](figures/05_saleprice_by_property_type.png)

---

## Step 6 — Supplementary (synthetic) data

Module 1's five extra columns (from [../data/final_data/train_v2.csv](../data/final_data/train_v2.csv)) were
profiled for plausibility and correlated with `SalePrice`. Two are **derived**, one is **macro**, two are
**simulated**:

| Feature | Type | r with `SalePrice` | Verdict / handoff |
|---|---|---|---|
| `DistanceToCenter` | simulated | **+0.58** | Strong, but a **proxy for `Neighborhood`** (Spearman **0.78** with neighborhood median) — cheap old downtown is *near* center, premium developments on the *edge*. **Collinear with location; use one, not both.** |
| `DaysOnMarket` | simulated | **+0.52** | Positive & non-linear (luxury homes listed longer). Days-on-market **as of the data snapshot** and **present in the test set** → known at prediction time, **keep it** (document the assumption + run an ablation). |
| `AgeAtSale` | derived | **−0.52** | Reconfirms the age effect (= insight #9); already a recommended derived feature. |
| `YearsSinceRemodel` | derived | **−0.51** | Same signal as `Age`; keep one. |
| `MortgageRate` | macro | **+0.03** | **~No cross-sectional signal** — a time variable (one rate per sale month, 6.7%→4.6% over 2006–2010, a function of `YrSold`/`MoSold`). Keep for macro/temporal framing only. |

**Plausibility:** all derived identities hold 100%, `MortgageRate` is one value per (year, month), no nulls;
`YearsSinceRemodel` has 1 mildly negative value (inherited Ames quirk, kept as-is).
→ **Handoff to Module 4/5:** the derived/distance features are largely **redundant with existing signals**
(`DistanceToCenter` ↔ `Neighborhood`; `Age`/`YearsSinceRemodel` ↔ years). `DaysOnMarket` is a **data-snapshot
value known at prediction time** (present in the test set) → **keep it, but document the assumption and run an
ablation** (with/without) to confirm it is not over-contributing. Full digest in
[outputs/eda_summary.json](outputs/eda_summary.json) → `synthetic`.
📊 [figures/06_supplementary_distributions.png](figures/06_supplementary_distributions.png) ·
[figures/06_supplementary_correlations.png](figures/06_supplementary_correlations.png) ·
[figures/06_supplementary_relationships.png](figures/06_supplementary_relationships.png)

---

## Handoff files

| File | For | Contents |
|------|-----|----------|
| [outputs/missing_report.csv](outputs/missing_report.csv) | **Module 3** | 19 columns × (`n_missing`, `pct_missing`, `na_meaning`) |
| [outputs/outlier_candidates.csv](outputs/outlier_candidates.csv) | **Module 3** | 45 candidates (2 leverage + 43 multivariate IQR) with `n_outlier_flags`, `flagged_features`, `reason` |
| [outputs/correlation_top.csv](outputs/correlation_top.csv) | **Module 4** | All features ranked by \|corr\| with `SalePrice` |
| [outputs/collinear_pairs.csv](outputs/collinear_pairs.csv) | **Module 4** | 10 predictor pairs with \|r\| ≥ 0.70 |
| [outputs/eda_summary.json](outputs/eda_summary.json) | all | Machine-readable digest of every number above |

---

## Recommendations rollup

**For Module 3 (cleaning)**
1. Fill "structural NA" columns with `"None"`/`0`; keep the rows.
2. Impute the 3 genuine-missing columns: `LotFrontage` (neighborhood median), `MasVnrArea` (0), `Electrical` (mode).
3. Remove the 2 leverage outliers (`Id 524`, `Id 1299`); review the other 43 IQR candidates but keep genuine luxury homes.

**For Module 4 (feature engineering)**
1. Model target = `log(SalePrice)`.
2. Ordinal-encode the quality/condition columns (`Ex..Po → 5..1`, absent → 0) — they are top predictors.
3. Resolve the 10 collinear pairs (keep one each) and avoid the `GrLivArea`/`1stFlrSF`/`2ndFlrSF` VIF trap.
4. Add derived features: `Age`, `YearsSinceRemod`, `TotalSF` (= `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`), `HasPool`.
5. Encode `Neighborhood` by median price (strong, 3.6× spread).
6. Supplementary columns (`data/final_data/train_v2.csv`): `DistanceToCenter` is collinear with `Neighborhood`
   (keep one); `MortgageRate` is a macro/time proxy only; **keep `DaysOnMarket`** (known at prediction time as a
   data-snapshot value, present in the test set) but **document the assumption and run an ablation**.

---

## Open items

- **Step 6 — supplementary data: ✅ done.** The days-on-market / macro-rate / distance-to-center columns from
  Module 1 have been profiled and correlated with `SalePrice` (see the Step 6 section above). Main takeaway:
  the strong synthetic signals are **collinear with existing features**; `DaysOnMarket` is a **data-snapshot
  value known at prediction time** (present in the test set) → **kept, with an assumption note + ablation
  recommendation**.

*See the full analysis and all charts in [01_eda.ipynb](01_eda.ipynb). Data dictionary: [DATA_DICTIONARY_EN.md](DATA_DICTIONARY_EN.md) / [DATA_DICTIONARY_VI.md](DATA_DICTIONARY_VI.md).*
