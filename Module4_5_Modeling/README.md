# Module 4-5 Modeling - House Price Prediction

This folder contains the Module 4 feature engineering and Module 5 model development pipeline for the Ames/Kaggle House Prices project.

Run commands from this folder, or prefix paths with `Module4_5_Modeling/` when running from the repository root.

## Data Source

Module 3 now provides cleaned data under `CleanData/`. The Module 4-5 pipeline treats these files as the primary source of truth:

- train: `CleanData/data-v2/processed/train_cleaned.csv`
- test/submission: `CleanData/data-v2/processed/test_cleaned.csv`

The cleaned train file contains `SalePrice`. The cleaned test file also contains `SalePrice` from the enriched Ames join, but the Kaggle submission writer only outputs `Id,SalePrice`.

The code does not modify or overwrite `CleanData/`.

## Time Context

`YrSold` and `MoSold` are valid prediction context because the business task predicts market price at a sale/search/valuation date. The feature pipeline keeps them by default and derives:

- `sale_year`
- `sale_month`
- `sale_quarter`
- `sale_ym_index`

For a later API/demo, the input should include either `valuation_date` or both `search_year` and `search_month` so the same temporal features can be derived.

## Install

```bash
pip install -r requirements.txt
```

From the repository root:

```bash
pip install -r Module4_5_Modeling/requirements.txt
```

`xgboost` is optional at runtime. If it is unavailable, the script trains the sklearn models and skips boosting gracefully.

## Run Feature Engineering

```bash
python scripts/run_feature_engineering.py
```

From the repository root:

```bash
python Module4_5_Modeling/scripts/run_feature_engineering.py
```

This writes engineered feature snapshots and feature lists under `outputs/`.

## Train Models

Default run using `CleanData/`, with derived, contextual, location, and time/macro features enabled:

```bash
python scripts/train_models.py
```

From the repository root:

```bash
python Module4_5_Modeling/scripts/train_models.py
```

Useful ablation-style switches:

```bash
python scripts/train_models.py --include-days-on-market false
python scripts/train_models.py --include-distance-to-center false
python scripts/train_models.py --include-neighborhood false
python scripts/train_models.py --include-time-features false --include-mortgage-rate false
```

Local raw-data fallback is only for smoke testing:

```bash
python scripts/train_models.py --allow-raw-fallback
```

## Models

The training script compares:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- XGBoost, if installed; otherwise LightGBM if installed; otherwise skipped

Training uses `log1p(SalePrice)`. Metrics and submissions are converted back to dollars with `expm1`.

## Ablations

`outputs/ablation_results.csv` compares:

- base cleaned features
- base plus derived features
- derived plus supplementary/contextual features
- with vs without `DaysOnMarket`
- `Neighborhood` only vs `DistanceToCenter` only vs both
- with vs without time/macro features, including `MortgageRate`

## Outputs

Training writes:

- `outputs/metrics_summary.csv`
- `outputs/cv_results.csv`
- `outputs/ablation_results.csv`
- `outputs/feature_importance.csv`
- `outputs/model_artifacts.csv`
- `outputs/tuning_results.csv`
- `outputs/vif_report.csv`
- `outputs/residual_summary.csv`
- `outputs/error_by_price_bucket.csv`
- `outputs/error_by_neighborhood.csv`
- `outputs/error_by_overallqual.csv`
- `figures/actual_vs_predicted.png`
- `figures/residual_plot.png`
- `figures/residuals_by_price_bucket.png`
- `models/best_model.pkl`
- `models/final_pipeline.pkl`
- `models/checkpoints/*_pipeline.pkl`
- `models/checkpoints/*_model.pkl`
- `submissions/kaggle_submission.csv`

`final_pipeline.pkl` is the selected best pipeline. Each file under `models/checkpoints/*_pipeline.pkl` is also a full reusable pipeline for that model family, including feature engineering, preprocessing, and the trained estimator. `Id` is preserved for submission but removed from model features.

## Inference Smoke Test

The deployment inference utility is:

```text
src/predict.py
```

Run the smoke test:

```bash
python src/smoke_test_inference.py
```

From the repository root:

```bash
python Module4_5_Modeling/src/smoke_test_inference.py
```

The inference path loads `models/final_pipeline.pkl`, applies the saved feature engineering and preprocessing, predicts on the log scale, and converts to USD with `expm1`.

## For Module 6-7 Team

1. Read `reports/HANDOFF_TO_MODULE_6_7.md` first.
2. Use `src/predict.py` for inference.
3. Do not manually reimplement feature engineering in the API.
4. Provide valuation/search date in the API/UI using `valuation_date` or `search_year` + `search_month`.
5. Log all prediction requests for monitoring.

Important handoff files:

- `reports/HANDOFF_TO_MODULE_6_7.md`
- `reports/MODEL_CARD_LASSO_V1.md`
- `reports/MODULE_4_FEATURE_ENGINEERING.md`
- `reports/MODULE_5_MODEL_DEVELOPMENT.md`
- `src/predict.py`
- `src/smoke_test_inference.py`
- `models/final_pipeline.pkl`
- `outputs/metrics_summary.csv`
- `outputs/residual_summary.csv`
- `outputs/vif_report.csv`
