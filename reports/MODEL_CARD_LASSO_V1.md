# Model Card - lasso_v1

## Model Summary

- Model name: `lasso_v1`
- Model type: Lasso Regression inside a full sklearn Pipeline
- Artifact: `models/final_pipeline.pkl`
- Target: `log1p(SalePrice)`
- Output: predicted sale price in USD after `expm1`

## Intended Use

Estimate expected house sale price at a provided sale/search/valuation date for the Ames/Kaggle-style real estate listing project.

## Not Intended For

- Legal appraisal
- Guaranteed transaction pricing
- Mortgage underwriting
- Direct use outside comparable data distributions without validation

## Training Data

Primary training file:

```text
CleanData/data-v2/processed/train_cleaned.csv
```

Module 3 removed the two high-leverage outliers and handled data cleaning. Module 4-5 did not modify `CleanData/`.

## Input Feature Categories

- Property size and structure: living area, basement area, garage, bathrooms, porch area
- Quality and condition: overall quality, kitchen quality, exterior quality, basement quality
- Location: `Neighborhood`, `DistanceToCenter`
- Time context: `YrSold`, `MoSold`, derived time fields
- Macro/context fields: `MortgageRate`, `DaysOnMarket`
- Derived features: age, years since remodel, total square footage, amenity flags

## Metrics

Current validation metrics:

- RMSE: 18,680.63
- MAE: 13,195.73
- MAPE: 0.0785
- R2: 0.9368
- Within +/-10% actual price: 0.7500

## Limitations

- Historical data may not represent current or future market regimes.
- `MortgageRate` and valuation/search date must be provided consistently.
- `DaysOnMarket` must be known as of valuation/search date, not after sale.
- Simulated/contextual features can introduce assumptions that should be disclosed.
- High-end or unusual properties may have larger residuals.

## Risks

- Overvaluation can mislead sellers or buyers.
- Undervaluation can reduce expected listing value.
- Stale macro/time context can degrade predictions.
- Neighborhood/location features may encode market disparities.

## Monitoring Recommendations

Monitor input drift, prediction drift, and realized error when actual sale prices become available. Key metrics:

- RMSE
- MAE
- MAPE
- percent within +/-10%
- residual by price bucket
- residual by `Neighborhood`
- residual by `OverallQual`
