# Module 4 - Feature Engineering

## Input Data

Primary data source: `CleanData/data-v2/processed/train_cleaned.csv` and `CleanData/data-v2/processed/test_cleaned.csv`.

Module 4 does not modify `CleanData/` and does not redo Module 3 cleaning. Remaining missing values are handled inside the sklearn pipeline for deployment safety.

## Target Handling

The supervised target is transformed with:

```python
y = np.log1p(SalePrice)
```

Model predictions are converted back to USD with:

```python
predicted_sale_price = np.expm1(predicted_log_price)
```

## Leakage Prevention

- `SalePrice` is removed from model inputs.
- `Id` is preserved only for submission/tracking and removed from model features.
- `price_per_sqft`, `price_per_m2`, and similar target-derived price ratio features are excluded because they would leak `SalePrice`.
- `DaysOnMarket` is treated as valid only if it is known as of the search/valuation date.

## Derived Features

The feature engineering transformer creates or preserves:

- `age_at_sale`
- `years_since_remodel`
- `total_sf`
- `total_bath`
- `total_porch_sf`
- `has_pool`
- `has_garage`
- `has_basement`
- `has_fireplace`
- `has_fence`
- quality-size interactions such as `qual_gr_liv_area` and `qual_total_sf`

Negative age/remodel values are flagged and clipped to zero inside the feature transformer when present.

## Time And Context Features

The project predicts price at a sale/search/valuation date. Therefore time context is retained:

- `YrSold` / `MoSold` are kept as `yr_sold` / `mo_sold`.
- Derived time features include `sale_quarter`; `sale_ym_index` is available in the full feature set.
- Deployment must provide `valuation_date` or `search_year` + `search_month`.
- `MortgageRate` is retained as a macro/time feature.
- `DistanceToCenter` is retained as a location proxy and compared against `Neighborhood` in ablations.

## Encoding And Scaling

- True ordinal quality/condition fields use ordered mappings, for example `NoFeature/None=0`, `Po=1`, `Fa=2`, `TA=3`, `Gd=4`, `Ex=5`.
- Nominal variables are one-hot encoded with `handle_unknown="ignore"`.
- `MSSubClass` is treated as categorical, not numeric.
- Numeric features are median-imputed; categorical features are most-frequent imputed.
- Linear models use `StandardScaler`; tree and boosted models do not require scaling.

## Multicollinearity

The project keeps a prediction-focused full feature set for tree/boosting models and a reduced linear feature set for linear/Ridge/Lasso models. The reduced set removes obvious exact-definition traps such as area component sums and duplicate time features.

VIF report: `outputs/vif_report.csv`.

Current VIF still shows high multicollinearity for engineered size/quality interaction groups, for example `total_sf`, `gr_liv_area`, `qual_total_sf`, and `qual_gr_liv_area`. Ridge and Lasso are appropriate because they regularize correlated predictors; Lasso also performs embedded feature selection.

Feature list files:

- `outputs/feature_list_full.csv`
- `outputs/feature_list_reduced_linear.csv`
