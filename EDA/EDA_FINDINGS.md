# EDA Findings — House Prices (Ames, Iowa)

**Module 2 · Branch `hacao` · Data Analyst**
Source: `data/train.csv` (1460 rows × 81 columns) · Notebook: [01_eda.ipynb](01_eda.ipynb)

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
| High-leverage outliers | **Id 524 & 1299** | Remove before modelling |

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

### 3. Two high-leverage outliers must be removed
`Id 524` (GrLivArea **4676** sqft, sold **\$184,750**) and `Id 1299` (GrLivArea **5642** sqft, sold **\$160,000**)
are the two largest homes in the data yet sold far below trend. Both have `OverallQual = 10`, are in
**Edwards**, and are **`Partial`** sales (unfinished when assessed) — which explains the low price. They distort
any linear fit.
→ **Handoff to Module 3:** drop or down-weight — [outputs/outlier_candidates.csv](outputs/outlier_candidates.csv).
📊 [figures/03_grlivarea_vs_saleprice.png](figures/03_grlivarea_vs_saleprice.png)

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

## Handoff files

| File | For | Contents |
|------|-----|----------|
| [outputs/missing_report.csv](outputs/missing_report.csv) | **Module 3** | 19 columns × (`n_missing`, `pct_missing`, `na_meaning`) |
| [outputs/outlier_candidates.csv](outputs/outlier_candidates.csv) | **Module 3** | Id 524 & 1299 with context + reason |
| [outputs/correlation_top.csv](outputs/correlation_top.csv) | **Module 4** | All features ranked by \|corr\| with `SalePrice` |
| [outputs/collinear_pairs.csv](outputs/collinear_pairs.csv) | **Module 4** | 10 predictor pairs with \|r\| ≥ 0.70 |
| [outputs/eda_summary.json](outputs/eda_summary.json) | all | Machine-readable digest of every number above |

---

## Recommendations rollup

**For Module 3 (cleaning)**
1. Fill "structural NA" columns with `"None"`/`0`; keep the rows.
2. Impute the 3 genuine-missing columns: `LotFrontage` (neighborhood median), `MasVnrArea` (0), `Electrical` (mode).
3. Remove/down-weight outliers `Id 524` and `Id 1299`.

**For Module 4 (feature engineering)**
1. Model target = `log(SalePrice)`.
2. Ordinal-encode the quality/condition columns (`Ex..Po → 5..1`, absent → 0) — they are top predictors.
3. Resolve the 10 collinear pairs (keep one each) and avoid the `GrLivArea`/`1stFlrSF`/`2ndFlrSF` VIF trap.
4. Add derived features: `Age`, `YearsSinceRemod`, `TotalSF` (= `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`), `HasPool`.
5. Encode `Neighborhood` by median price (strong, 3.6× spread).

---

## Open items

- **Step 6 — synthetic data (pending Module 1).** Once the amenity-score / days-on-market / macro-indicator
  data is delivered, add a section to (a) check the plausibility of those distributions and (b) plot
  `SalePrice` vs distance-to-amenities, then re-run [01_eda.ipynb](01_eda.ipynb) and update this summary.

*See the full analysis and all charts in [01_eda.ipynb](01_eda.ipynb). Data dictionary: [DATA_DICTIONARY_EN.md](DATA_DICTIONARY_EN.md) / [DATA_DICTIONARY_VI.md](DATA_DICTIONARY_VI.md).*
