# Handoff To Module 6-7

## 1. Final Model Summary

Final model: **Lasso Regression**.

Model version: `lasso_v1`.

The model predicts expected sale price at a provided sale/search/valuation date. It was trained on `log1p(SalePrice)` and inference converts predictions back to USD with `np.expm1`.

## 2. Exact Artifact Paths

Core artifacts:

- `Module4_5_Modeling/models/final_pipeline.pkl`
- `Module4_5_Modeling/models/best_model.pkl`
- `Module4_5_Modeling/models/preprocessing_pipeline.pkl`
- `Module4_5_Modeling/models/checkpoints/lasso_regression_pipeline.pkl`
- `Module4_5_Modeling/outputs/metrics_summary.csv`
- `Module4_5_Modeling/outputs/residual_summary.csv`
- `Module4_5_Modeling/outputs/feature_importance.csv`
- `Module4_5_Modeling/outputs/vif_report.csv`

Inference code:

- `Module4_5_Modeling/src/predict.py`
- `Module4_5_Modeling/src/smoke_test_inference.py`

Secondary Kaggle artifact:

- `Module4_5_Modeling/submissions/kaggle_submission.csv`

## 3. Actual Artifact Design

This repo uses **Case A**:

```text
Module4_5_Modeling/models/final_pipeline.pkl = full sklearn Pipeline
feature engineering -> preprocessing -> final Lasso model
```

Inference should therefore use:

```python
pipeline = joblib.load("Module4_5_Modeling/models/final_pipeline.pkl")
pred_log = pipeline.predict(input_df)
pred_price = np.expm1(pred_log)
```

Do not refit encoders, scalers, imputers, or feature engineering logic in deployment.

## 4. How To Load And Predict

Recommended Module 6 usage:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("Module4_5_Modeling/src").resolve()))
from predict import load_artifacts, predict_price

artifacts = load_artifacts()
result = predict_price(raw_input, artifacts=artifacts)
```

The returned single-record object:

```json
{
  "predicted_sale_price": 245000.0,
  "model_version": "lasso_v1",
  "target_transform": "log1p",
  "inverse_transform": "expm1",
  "approx_validation_mape": 0.0785,
  "notes": "Prediction is for the provided valuation/search date context."
}
```

For batch inference, `predict_price` returns a DataFrame with at least:

- `predicted_sale_price`
- `model_version`

## 5. API/UI Input Schema

The API/UI should accept raw property fields. Do not manually recreate feature engineering in the API. Call `src.predict.predict_price`.

Recommended important fields:

- `Neighborhood`
- `OverallQual`
- `OverallCond`
- `GrLivArea`
- `TotalBsmtSF`
- `GarageCars`
- `GarageArea`
- `FullBath`
- `HalfBath`
- `BedroomAbvGr`
- `KitchenQual`
- `YearBuilt`
- `YearRemodAdd`
- `SaleCondition`
- `MortgageRate`
- `DaysOnMarket`
- `DistanceToCenter`

The inference utility fills missing non-critical training columns from cleaned training defaults. It still requires core fields and date context.

## 6. Required Date/Search Fields

Module 6 must provide one of:

```text
valuation_date
```

or:

```text
search_year
search_month
```

Reason:

- Training used `YrSold` and `MoSold` as sale-date context.
- Deployment should map valuation/search date to the same context.
- The inference utility maps `valuation_date` or `search_year/search_month` to `YrSold`/`MoSold`.
- It does not silently default to the current date.

## 7. Prediction Output Format

Recommended API response:

```json
{
  "predicted_sale_price": 245000.0,
  "model_version": "lasso_v1",
  "target_transform": "log1p",
  "inverse_transform": "expm1",
  "approx_validation_mape": 0.0785,
  "notes": "Prediction is for the provided valuation/search date context."
}
```

## 8. Monitoring Fields To Log

Log every prediction:

- `request_id`
- `timestamp`
- `model_version`
- `raw_input_json`
- `engineered_feature_snapshot` if available
- `predicted_sale_price`
- `valuation_year`
- `valuation_month`
- `MortgageRate`
- `DaysOnMarket`
- `Neighborhood`
- `OverallQual`
- `GrLivArea`
- `TotalSF` / `total_sf` if available
- `AgeAtSale` / `age_at_sale` if available
- `actual_sale_price` if later known
- `residual` if actual is available
- `absolute_error` if actual is available
- `absolute_percentage_error` if actual is available

## 9. Drift And Performance Metrics

Input drift:

- `OverallQual`
- `GrLivArea`
- `TotalSF` / `total_sf`
- `AgeAtSale` / `age_at_sale`
- `YearsSinceRemodel` / `years_since_remodel`
- `Neighborhood`
- `GarageCars`
- `TotalBath` / `total_bath`
- `MortgageRate`
- `DaysOnMarket`
- `DistanceToCenter`

Prediction drift:

- predicted sale price distribution
- predicted sale price by month
- predicted sale price by `Neighborhood`
- predicted sale price by `OverallQual`
- share of very low/high predictions

Performance when actual sale price is available:

- RMSE
- MAE
- MAPE
- percent within +/-10% actual price
- mean residual
- median residual
- residual by price bucket
- residual by `Neighborhood`
- residual by `OverallQual`

Residual definition:

```text
residual = actual_sale_price - predicted_sale_price
```

Interpretation:

- `residual > 0`: model underpredicts.
- `residual < 0`: model overpredicts.

## 10. Retraining Triggers

Suggested retraining triggers:

- MAPE rises above 10-12%.
- Percent within +/-10% drops meaningfully below validation level.
- Prediction distribution shifts strongly from validation distribution.
- Feature drift appears in `Neighborhood`, `OverallQual`, `GrLivArea`, `MortgageRate`, or `DaysOnMarket`.
- New actual sale prices accumulate for a newer market period.

## 11. Known Limitations And Warnings

- This is not a legal appraisal model.
- `DaysOnMarket` must reflect days active as of valuation/search date, not future information known only after sale.
- `MortgageRate` should match the valuation/search month.
- `DistanceToCenter` is contextual/simulated and should be disclosed.
- The model is trained on Ames/Kaggle-style data and may not generalize to other cities without retraining.

## 12. Files Module 6-7 Should Read First

1. `Module4_5_Modeling/reports/HANDOFF_TO_MODULE_6_7.md`
2. `Module4_5_Modeling/reports/MODEL_CARD_LASSO_V1.md`
3. `Module4_5_Modeling/src/predict.py`
4. `Module4_5_Modeling/src/smoke_test_inference.py`
5. `Module4_5_Modeling/models/final_pipeline.pkl`
6. `Module4_5_Modeling/outputs/metrics_summary.csv`
7. `Module4_5_Modeling/outputs/residual_summary.csv`
8. `Module4_5_Modeling/outputs/vif_report.csv`
