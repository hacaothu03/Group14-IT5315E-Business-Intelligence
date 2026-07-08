# Module 5 - Model Development

## Models Trained

The pipeline trained and compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- XGBoost
- Tuned Random Forest using randomized search
- Tuned XGBoost using randomized search

All models use `log1p(SalePrice)` as the training target and report metrics on the original USD scale after `expm1`.

## Validation

Validation strategy:

- 80/20 train-validation split with random seed 42.
- 5-fold cross-validation for baseline model comparison.
- Light `RandomizedSearchCV` with 3-fold CV for Random Forest and XGBoost tuning.

Detailed outputs:

- `outputs/metrics_summary.csv`
- `outputs/cv_results.csv`
- `outputs/tuning_results.csv`

## Metrics Summary

Current validation metrics:

| Model | RMSE | MAE | MAPE | R2 |
|---|---:|---:|---:|---:|
| Lasso Regression | 18,680.63 | 13,195.73 | 0.0785 | 0.9368 |
| Ridge Regression | 19,033.64 | 13,465.43 | 0.0816 | 0.9344 |
| XGBoost | 19,832.03 | 14,132.98 | 0.0852 | 0.9288 |
| Tuned XGBoost | 19,932.67 | 14,176.87 | 0.0856 | 0.9281 |
| Linear Regression | 21,604.00 | 14,700.70 | 0.0879 | 0.9155 |
| Tuned Random Forest | 22,182.82 | 15,041.97 | 0.0960 | 0.9109 |
| Random Forest | 22,528.81 | 15,302.28 | 0.0979 | 0.9081 |

## Final Model Selection

Final model: **Lasso Regression** (`lasso_v1`).

Reasons:

- Lowest validation RMSE among trained models.
- Low validation MAPE, approximately 7.85%.
- Small generalization gap compared with Random Forest and XGBoost.
- Regularization helps with correlated engineered features.
- More interpretable and deployable than overfit tree/boosting variants.

Final artifacts:

- `models/final_pipeline.pkl`
- `models/best_model.pkl`
- `models/checkpoints/lasso_regression_pipeline.pkl`

## Ablation Findings

Ablation output: `outputs/ablation_results.csv`.

Main findings:

- Derived features improve performance versus base cleaned features.
- `Neighborhood` remains stronger than `DistanceToCenter` alone.
- Adding `DistanceToCenter` to `Neighborhood` does not materially improve Ridge ablation results.
- Removing `DaysOnMarket` slightly improves Ridge ablation, so it should be treated carefully and documented as available only at valuation/search time.
- Removing time and mortgage features does not strongly degrade validation metrics, but these features remain valid context for price-at-date prediction.

## Residual Analysis

Residual outputs:

- `outputs/residual_summary.csv`
- `outputs/error_by_price_bucket.csv`
- `outputs/error_by_neighborhood.csv`
- `outputs/error_by_overallqual.csv`

Figures:

- `figures/actual_vs_predicted.png`
- `figures/residual_plot.png`
- `figures/residual_distribution.png`
- `figures/residuals_by_price_bucket.png`

Residual definition:

```text
residual = actual_sale_price - predicted_sale_price
```

Positive residual means underprediction. Negative residual means overprediction.

## Known Limitations

- Data is historical Ames/Kaggle data; model may not generalize to different regions without retraining.
- `DaysOnMarket` must not be a future-known post-sale value.
- `DistanceToCenter` is partly simulated/contextual and should be explained in the report.
- The model is not a legal appraisal tool.
