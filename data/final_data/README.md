# Enriched Dataset — Version 2 (`train_v2.csv` / `test_v2.csv`)

Documentation for the Kaggle dataset enriched with extra columns (version 2).

---

## 1. Overview

`train_v2.csv` and `test_v2.csv` are the **original Kaggle files kept fully intact**, with a few supplementary columns simply **appended**. In addition, `test_v2.csv` receives the target column `SalePrice` (which is hidden in the Kaggle competition file).

| File | Rows | Columns | = Original + Added |
|---|---|---|---|
| `train_v2.csv` | 1,460 | 86 | 81 (Kaggle) + 5 |
| `test_v2.csv`  | 1,459 | 86 | 80 (Kaggle) + 5 + `SalePrice` |

---

## 2. Data lineage

- **Kaggle originals:** `../kaggle_data/train.csv` (1460×81), `../kaggle_data/test.csv` (1459×80)
  — *House Prices: Advanced Regression Techniques*, columns like `MSSubClass`, key `Id` 1…2919.
- **Source of added columns:** `../housing_initial_data.csv` (2930×87)
  — the **full Ames Housing** set (keys `Order`/`PID`, spaced column names like `MS SubClass`), already carrying the 5 supplementary columns and a `SalePrice` for every row.

`train_v2`/`test_v2` are **not** a split of `housing_initial_data.csv`; they are the separate Kaggle variant, but **every row exists in** the full Ames file, so the added columns are transferred from there.

---

## 3. Added columns

| Column | Meaning | Type | Range |
|---|---|---|---|
| `AgeAtSale` | House age at sale = `YrSold − YearBuilt` | Derived | ~ 0–136 (*) |
| `YearsSinceRemodel` | Years since last remodel = `YrSold − YearRemodAdd` | Derived | ~ −2–60 (*) |
| `MortgageRate` | 30-yr mortgage rate (%) at sale month | Macro (exogenous) | 4.564 – 6.7625 |
| `DaysOnMarket` | Days listed until sold | Simulated | 21 – 168 |
| `DistanceToCenter` | Distance to city center (km) | Simulated | 0.1 – 5.85 |
| `SalePrice` *(test only)* | Sale price (USD) | Target | 12,789 – 615,000 |

**(*) Note:** `AgeAtSale` and `YearsSinceRemodel` may be **negative** for a few rows — an **inherent quirk of the original Ames data** (remodel/build year later than sale year), preserved as-is, not a join error.

- `MortgageRate`: exactly **one** rate per (`YrSold`, `MoSold`) pair, trending down from ~6.7% (2006) to ~4.5% (2010), matching real U.S. rates.

---

## 4. How the join was done

Since the Kaggle files lack `PID`, the added columns were joined by matching the **79 common feature columns** between the two datasets (after normalizing column names and numeric types):

1. **Exact match on 79 columns** → 1,460/1,460 for train and 1,440/1,459 for test.
2. **19 remaining test rows**: Kaggle had replaced some **rare category values** with blanks (NA). These were resolved by a **unique nearest match** — differing only in the blanked cell — so the corresponding house is still identified exactly.
3. **2 identical-feature twins** were disambiguated by `SalePrice`.

> **Important:** Cells that Kaggle left blank (NA) **remain blank** in `_v2` — the rare Ames values were **not** written back. All original columns are identical to Kaggle cell-by-cell.

---

## 5. Validation

| Check | Result |
|---|---|
| Original Kaggle columns unchanged, cell-by-cell (train & test) | ✅ Identical |
| Joined train `SalePrice` == original `SalePrice` | ✅ 1460 / 1460 |
| `AgeAtSale == YrSold − YearBuilt` | ✅ 100% |
| `YearsSinceRemodel == YrSold − YearRemodAdd` | ✅ 100% |
| `MortgageRate`: one rate per (year, month) | ✅ |
| 1:1 mapping, no train/test overlap | ✅ overlap = 0 |
| Nulls in added columns (incl. test `SalePrice`) | ✅ 0 null |

---

## 6. Usage notes

- Use `../kaggle_data/` for the untouched Kaggle data; use `final_data/*_v2.csv` when you need the added features and the test `SalePrice` label (e.g. for offline evaluation).
- ⚠️ `DaysOnMarket` and `DistanceToCenter` are **simulated** — state this clearly in any report.

---

*Updated: 2026-07-06 · Group 14 · IT5315E Business Intelligence*
