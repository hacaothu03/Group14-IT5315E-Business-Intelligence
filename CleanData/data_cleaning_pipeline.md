# Data Cleansing Pipeline Walkthrough

This document explains the data cleansing pipeline implemented in `src/data_cleaning.py` for the Kaggle House Prices: Advanced Regression Techniques dataset. The pipeline prepares the original Kaggle files, or the augmented v2 files when available, for later EDA, feature engineering, and model training with baseline regressors such as Linear Regression and Decision Tree models.

The goal of this stage is not to maximize model accuracy directly. The goal is to make the raw Ames Housing data more consistent, auditable, and robust so later modeling steps do not confuse domain-specific missing values, data-entry gaps, and true structural absence.

## 1. Inputs And Outputs

### Input files

The pipeline reads files from the `data/` directory. It automatically prefers augmented v2 files when they are present:

- `data/train_v2.csv` if available, otherwise `data/train.csv`.
- `data/test_v2.csv` if available, otherwise `data/test.csv`.

The original Kaggle files are:

- `data/train.csv`: training data with `SalePrice`.
- `data/test.csv`: Kaggle test data without `SalePrice`.

The augmented v2 files add business/context features documented in `documents/augmented_features.md`:

- `AgeAtSale`
- `YearsSinceRemodel`
- `MortgageRate`
- `DaysOnMarket`
- `DistanceToCenter`
- `SalePrice` in `test_v2.csv` for offline evaluation, if that file is present.

The original raw Kaggle dataset contains:

- Train shape: `1460` rows and `81` columns.
- Test shape: `1459` rows and `80` columns.
- No duplicate `Id` values in either file.

The documented v2 dataset contains:

- `train_v2.csv`: `1460` rows and `86` columns.
- `test_v2.csv`: `1459` rows and `86` columns, including offline-evaluation `SalePrice`.

### Output files

Running the pipeline writes the following files to `data/processed/`:

- `train_cleaned.csv`: cleaned training dataset.
- `test_cleaned.csv`: cleaned test dataset.
- `cleaning_report.json`: machine-readable audit report with shapes, missing values, imputations, and removed outliers.
- `cleaning_summary.md`: short human-readable summary of the cleaning results.

After cleaning:

- Clean train keeps all selected input features, plus deterministic augmented fields if they need to be derived.
- Clean test keeps all selected input features, plus deterministic or schema-alignment augmented fields when needed.
- Missing values after normal CSV read-back: `0` in both train and test.
- Duplicate `Id` values: `0` in both train and test.

After cleaning with augmented v2 inputs, the cleaned files keep the augmented columns. In the current workspace, `train_v2.csv` exists but `test_v2.csv` is not present, so the pipeline can derive or align the test-side augmented columns from `test.csv` and records that handling in `cleaning_report.json`.

The training set has two fewer rows because two documented high-leverage outliers are removed. The test set is never row-filtered because Kaggle submissions must preserve the original test-row alignment.

## 2. How To Run

From the project root, run:

```powershell
python src/data_cleaning.py --raw-dir data --output-dir data/processed
```

The script also works with defaults:

```powershell
python src/data_cleaning.py
```

Arguments:

- `--raw-dir`: directory containing `train.csv` and `test.csv`. Default: `data`.
- `--output-dir`: directory where cleaned files and reports are written. Default: `data/processed`.
- `--train-file`: optional explicit train file path, for example `data/train_v2.csv`.
- `--test-file`: optional explicit test file path, for example `data/test_v2.csv`.

## 3. Pipeline Design Principles

### Preserve domain meaning

In the Kaggle Ames dataset, many missing values are not unknown values. They mean that the house does not have a feature. For example:

- Missing `Alley` means no alley access.
- Missing basement quality/type fields often means no basement.
- Missing garage fields often means no garage.
- Missing `PoolQC` usually means no pool.
- Missing `FireplaceQu` means no fireplace.

Replacing all missing values with medians or modes would erase this information. Therefore, the pipeline separates domain-specific absence from true missing data-entry gaps.

### Keep cleaning separate from feature engineering

The cleaned outputs are not one-hot encoded, scaled, or log-transformed. Those transformations belong to the later feature-engineering/modeling pipeline. This keeps the cleaned CSV files reusable for both linear models and tree-based models.

### Preserve augmented feature lineage

The augmented columns are treated as already-defined business/context features, not as arbitrary missing values. The pipeline preserves them when v2 files exist, derives deterministic fields when possible, and writes any fallback handling into the audit report.

### Avoid broad hidden outlier clipping

The pipeline does not cap every extreme value. It removes only two clearly documented training outliers where `GrLivArea` is extremely large but `SalePrice` is unusually low. Other extreme observations are kept for EDA and modeling decisions.

## 4. Step-By-Step Implementation

### Step 1: Load raw or augmented data

Implemented by `_load_raw_data()`.

The pipeline automatically chooses `train_v2.csv` and `test_v2.csv` when they exist. If a v2 file is missing, it falls back to the matching original Kaggle file. Explicit paths can also be supplied with `--train-file` and `--test-file`.

It raises a `FileNotFoundError` if the selected train or test file is missing, making failures explicit instead of silently producing incomplete outputs.

### Step 2: Normalize categorical strings

Implemented by `_normalize_categorical_strings()`.

For every text column, the pipeline:

- Strips leading and trailing whitespace.
- Converts empty strings to missing values.

This prevents category duplication caused by formatting differences, such as `"RL"` versus `" RL "`, and prevents blank strings from being treated as real categories.

### Step 3: Remove duplicate IDs

Implemented inside `run_cleaning_pipeline()`.

The pipeline drops duplicate rows using `Id` as the unique key:

```python
raw_train = raw_train.drop_duplicates(subset=["Id"]).copy()
raw_test = raw_test.drop_duplicates(subset=["Id"]).copy()
```

The current raw data already has no duplicate IDs, but this check protects the pipeline if the dataset is merged, regenerated, or manually edited later.

### Step 4: Separate target columns from features

Implemented inside `run_cleaning_pipeline()`.

The training target `SalePrice` is temporarily separated from the training features. Cleaning rules are applied to feature columns first, then `SalePrice` is attached again before outlier handling.

If `test_v2.csv` is used and contains `SalePrice`, the test target is also separated during feature cleaning and attached again to the cleaned test output. This supports offline evaluation while preventing target leakage into feature imputation.

This separation keeps train and test feature schemas consistent.

### Step 5: Preserve or create augmented features

Implemented by `_ensure_augmented_features()` and `_fill_by_month_rate()`.

The v2 augmentation adds:

| Column | Handling in the cleaning pipeline |
| --- | --- |
| `AgeAtSale` | Preserved if present; otherwise derived as `YrSold - YearBuilt`. |
| `YearsSinceRemodel` | Preserved if present; otherwise derived as `YrSold - YearRemodAdd`. |
| `MortgageRate` | Preserved if present; if missing in test, filled from train by `(YrSold, MoSold)` mortgage-rate mapping, with train median as fallback. |
| `DaysOnMarket` | Preserved if present; if `test_v2.csv` is missing but train has the column, test is filled with the train median for schema alignment. |
| `DistanceToCenter` | Preserved if present; if `test_v2.csv` is missing but train has the column, test is filled with the train median for schema alignment. |

This behavior handles the current project state where `train_v2.csv` is available but `test_v2.csv` is not. Deterministic features are recomputed from existing Kaggle columns. Simulated features cannot be exactly reconstructed from `test.csv`, so median fallback is used only to keep the model input schema aligned. The fallback is documented in `cleaning_report.json`.

The pipeline also validates augmented fields:

- `AgeAtSale == YrSold - YearBuilt`
- `YearsSinceRemodel == YrSold - YearRemodAdd`
- `MortgageRate` has one rate per `(YrSold, MoSold)` pair
- negative age/remodel counts are reported, not automatically removed

Negative `AgeAtSale` or `YearsSinceRemodel` values are preserved because the augmentation document identifies these as quirks of the source Ames data rather than join errors.

### Step 6: Impute `LotFrontage` by neighborhood

Implemented by `_impute_lot_frontage()`.

`LotFrontage` is the linear feet of street connected to a property. It has many missing values:

- Train: `259`
- Test: `227`

The value is location-dependent, so the pipeline fills it using:

1. Median `LotFrontage` within the same `Neighborhood`.
2. Global median `LotFrontage` if a neighborhood median is unavailable.

The median is used instead of the mean because frontage can be skewed by very large lots.

### Step 7: Fill domain-specific absence values

Implemented by `_impute_domain_absence()`.

The pipeline converts missing categorical values that mean feature absence into the explicit label `NoFeature`.

Affected categorical columns:

- `Alley`
- `BsmtQual`
- `BsmtCond`
- `BsmtExposure`
- `BsmtFinType1`
- `BsmtFinType2`
- `FireplaceQu`
- `GarageType`
- `GarageFinish`
- `GarageQual`
- `GarageCond`
- `PoolQC`
- `Fence`
- `MiscFeature`
- `MasVnrType`

The pipeline uses `NoFeature` instead of `None` because pandas treats strings such as `"None"` as missing values when CSVs are read with default settings. Using `NoFeature` keeps the cleaned CSV files stable when they are loaded in later notebooks.

Matching numeric fields are filled with `0` where missing means the structure does not exist:

- `MasVnrArea`
- `BsmtFinSF1`
- `BsmtFinSF2`
- `BsmtUnfSF`
- `TotalBsmtSF`
- `BsmtFullBath`
- `BsmtHalfBath`
- `GarageCars`
- `GarageArea`
- `GarageYrBlt`

For `GarageYrBlt`, `0` means there is no garage year because there is no garage. Later feature engineering should handle this carefully, for example by creating `HasGarage` or converting the value into a garage-age feature only when a garage exists.

### Step 8: Impute true missing categorical values

Implemented by `_impute_true_missing()`.

Some missing values are data-entry gaps, not meaningful feature absence. These are filled using training-set defaults so train and test transformations stay consistent.

Mode-imputed categorical columns:

| Column | Fill value |
| --- | --- |
| `MSZoning` | `RL` |
| `Utilities` | `AllPub` |
| `Exterior1st` | `VinylSd` |
| `Exterior2nd` | `VinylSd` |
| `Electrical` | `SBrkr` |
| `KitchenQual` | `TA` |
| `Functional` | `Typ` |
| `SaleType` | `WD` |

`Functional` is filled with `Typ` because the data dictionary defines it as the typical home functionality category, and the Kaggle competition is commonly handled this way unless there is evidence of a functional defect.

### Step 9: Impute remaining numeric gaps

Implemented by `_impute_true_missing()`.

After domain absence handling, any remaining numeric missing values are filled with the training-set median for that column. The `Id` column is excluded because it is an identifier, not a model feature.

In the current generated report, no additional numeric median imputations remain after the earlier domain and `LotFrontage` handling steps.

### Step 10: Normalize domain relationships

Implemented by `_normalize_domain_rules()`.

This step makes related columns agree with one another.

Masonry veneer:

- If `MasVnrArea` is `0`, then `MasVnrType` is set to `NoFeature`.
- If `MasVnrType` is `NoFeature`, missing `MasVnrArea` is treated as `0`.

Fireplaces:

- If `Fireplaces` is `0`, then `FireplaceQu` is set to `NoFeature`.

Pools:

- If `PoolArea` is `0`, then `PoolQC` is set to `NoFeature`.

Garages:

- If `GarageType` is `NoFeature`, then `GarageFinish`, `GarageQual`, and `GarageCond` are set to `NoFeature`.
- If `GarageType` is `NoFeature`, then `GarageYrBlt`, `GarageCars`, and `GarageArea` are set to `0`.

Basements:

- If basement area columns sum to `0`, then basement quality, condition, exposure, and finish-type columns are set to `NoFeature`.

Year consistency:

- If `GarageYrBlt` is later than `YrSold`, it is replaced with `YearBuilt`.
- `YearBuilt` and `YearRemodAdd` are not rewritten. This preserves the date lineage needed by `AgeAtSale` and `YearsSinceRemodel`.

Categorical dwelling type:

- `MSSubClass` is converted from numeric to string. Although its values are numbers, the data dictionary defines them as dwelling-type codes, not continuous numeric quantities. Treating it as categorical avoids false ordinal assumptions in later models.

### Step 11: Remove documented training outliers

Implemented by `_remove_training_outliers()`.

The pipeline removes training rows matching:

```text
GrLivArea > 4000 and SalePrice < 300000
```

Removed row IDs:

- `524`
- `1299`

Reasoning:

- `GrLivArea` has a strong relationship with `SalePrice`.
- These two rows have very large living areas but unusually low sale prices.
- They can heavily distort simple models, especially Linear Regression.
- The same rule is not applied to the test set because the test set has no target and must preserve all Kaggle IDs.

## 5. Missing Data Summary

Before cleaning, the largest missing-value groups were mostly domain absence fields:

| Column | Train missing | Test missing | Handling |
| --- | ---: | ---: | --- |
| `PoolQC` | 1453 | 1456 | `NoFeature` |
| `MiscFeature` | 1406 | 1408 | `NoFeature` |
| `Alley` | 1369 | 1352 | `NoFeature` |
| `Fence` | 1179 | 1169 | `NoFeature` |
| `MasVnrType` | 872 | 894 | `NoFeature` / consistency rule |
| `FireplaceQu` | 690 | 730 | `NoFeature` |
| `LotFrontage` | 259 | 227 | Neighborhood median |
| Garage fields | around 81 | around 76-78 | `NoFeature` / `0` |
| Basement fields | around 37-38 | around 42-45 | `NoFeature` / `0` |

After cleaning, both cleaned CSVs contain no missing values when read back with default pandas settings.

## 6. Augmented Feature Handling Summary

The augmented features come from `train_v2.csv` / `test_v2.csv` when those files are available. They are handled as follows:

- `AgeAtSale` and `YearsSinceRemodel` are deterministic derived features. If a selected input file does not contain them, they are recreated from `YrSold`, `YearBuilt`, and `YearRemodAdd`.
- `MortgageRate` is a macro feature with one value per sale year/month. If it is missing from test data, the pipeline maps rates from the training data by `(YrSold, MoSold)` and falls back to the train median rate only for unseen year/month pairs.
- `DaysOnMarket` and `DistanceToCenter` are simulated/business-context features. If `test_v2.csv` is missing, they cannot be exactly recovered from `test.csv`, so the pipeline fills them with train medians for schema alignment and records those fill values in the report.
- If `test_v2.csv` is present and includes `SalePrice`, the cleaned test output keeps it for offline evaluation.

The audit report contains an `augmented_features` section with:

- `available_columns`: augmented columns found or created in the cleaned outputs.
- `handling`: derivation or fallback actions taken by the pipeline.
- `validation`: mismatch counts for deterministic derived features and mortgage-rate consistency checks.

## 7. Why This Improves Modeling Robustness

For Linear Regression:

- Explicit absence labels prevent row drops caused by missing values.
- Removing the two high-leverage `GrLivArea` outliers reduces slope distortion.
- Converting `MSSubClass` to categorical avoids treating dwelling codes as a continuous scale.
- Median imputation reduces the effect of skewed numeric distributions.
- Augmented numeric features provide interpretable age, macro-rate, listing-duration, and location-distance signals.

For Decision Tree models:

- Consistent absence categories allow trees to split on meaningful structural absence, such as no garage or no basement.
- Outlier removal reduces unstable branches around rare, suspicious records.
- Keeping unscaled numeric features is acceptable because tree splits are not sensitive to feature scale.

For both model families:

- Train and test receive the same cleaning logic.
- The target `SalePrice` is not used to compute imputation values.
- Augmented test labels, if available, are preserved only as targets for offline evaluation.
- The output is reproducible and auditable through JSON and Markdown reports.

## 8. Implementation Map

Main file:

- `src/data_cleaning.py`

Important functions:

- `_load_raw_data()`: loads raw train and test files.
- `_normalize_categorical_strings()`: trims text fields and normalizes blanks.
- `_ensure_augmented_features()`: preserves, derives, or aligns augmented v2 features.
- `_fill_by_month_rate()`: maps missing test mortgage rates from train by sale year/month.
- `_impute_lot_frontage()`: fills frontage by neighborhood median.
- `_impute_domain_absence()`: handles meaningful feature absence.
- `_impute_true_missing()`: fills remaining true missing values.
- `_normalize_domain_rules()`: enforces cross-column consistency.
- `_validate_augmented_features()`: records consistency checks for augmented fields.
- `_remove_training_outliers()`: removes documented training outliers.
- `_build_report()`: creates the audit report.
- `run_cleaning_pipeline()`: orchestrates the full pipeline.

## 9. Validation Checklist

After running the pipeline, these checks should pass:

- With original Kaggle-only inputs, `train_cleaned.csv` has `1458` rows and includes any deterministic augmented columns created by the current pipeline.
- With `train_v2.csv`, `train_cleaned.csv` keeps the five augmented columns and has `1458` rows after outlier removal.
- With `test_v2.csv`, `test_cleaned.csv` keeps the augmented columns and the offline-evaluation `SalePrice` target.
- If only `test.csv` is available, deterministic augmented columns are derived and simulated augmented columns are median-filled for schema alignment.
- Both cleaned files have `0` missing values after normal CSV read-back.
- Both cleaned files have `0` duplicate `Id` values.
- Training IDs `524` and `1299` are absent.
- Test IDs remain unchanged from `1461` to `2919`.
- `NoFeature` appears as a real category, not as a missing value.
- `cleaning_report.json` documents the input files and augmented-feature handling.

Example validation code:

```python
import pandas as pd

train = pd.read_csv("data/processed/train_cleaned.csv")
test = pd.read_csv("data/processed/test_cleaned.csv")

print(train.shape)
print(test.shape)
print(train.isna().sum().sum())
print(test.isna().sum().sum())
print(train["Id"].duplicated().sum())
print(test["Id"].duplicated().sum())
print(sorted(set([524, 1299]) & set(train["Id"].astype(int))))
print([c for c in ["AgeAtSale", "YearsSinceRemodel", "MortgageRate", "DaysOnMarket", "DistanceToCenter"] if c in train.columns])
```

Expected core output:

```text
(1458, ...)
(1459, ...)
0
0
0
0
[]
['AgeAtSale', 'YearsSinceRemodel', ...]
```

## 10. Limitations And Next Steps

This pipeline intentionally stops before feature engineering. Recommended next steps for the modeling stage include:

- One-hot encoding nominal categorical variables for Linear Regression.
- Ordinal encoding quality-related fields where the category order is meaningful.
- Creating flags such as `HasGarage`, `HasBasement`, `HasPool`, and `HasFireplace`.
- Creating age features such as `HouseAge`, `RemodAge`, and `GarageAge`.
- Deciding whether to use the provided `AgeAtSale` and `YearsSinceRemodel` directly or replace them with a more complete age-feature family.
- Considering a log transform of `SalePrice` for linear models.
- Checking skewed numeric features such as `LotArea`, `GrLivArea`, and basement/porch area fields.
- Treating median-filled `DaysOnMarket` and `DistanceToCenter` values cautiously when `test_v2.csv` is absent, because those simulated fields cannot be exactly reconstructed from original Kaggle test data.
- Running cross-validation to compare raw versus cleaned data performance.

These later steps should be implemented in a separate feature-engineering or modeling pipeline so the data-cleaning stage remains transparent and reusable.
