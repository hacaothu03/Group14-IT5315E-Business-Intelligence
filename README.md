# House Price Prediction - Modules 4 and 5

This repository contains the Module 4 feature engineering and Module 5 model development pipeline for the Kaggle Ames House Prices project.

## Current Inputs

The pipeline looks for training data in this order:

1. `data/processed/train_cleaned.csv`
2. `data/train_cleaned.csv`
3. `CLEANED_TRAIN_PATH`
4. `/kaggle/input/**/train_cleaned.csv`

If no cleaned file exists, training stops with a clear Module 3 message. For local smoke tests only, pass `--allow-raw-fallback` to use `data/train.csv`.

Test/submission data is resolved from `data/processed/test_cleaned.csv`, `data/test_cleaned.csv`, `data/test.csv`, or Kaggle input `test.csv`.

## Install

```bash
pip install -r requirements.txt
```

`xgboost` is optional at runtime. If it is unavailable, the script trains the sklearn models and logs that boosting was skipped.

## Run Feature Engineering

```bash
python scripts/run_feature_engineering.py --cleaned-train data/processed/train_cleaned.csv
```

Local smoke test before Module 3 delivers cleaned data:

```bash
python scripts/run_feature_engineering.py --allow-raw-fallback
```

## Train Models

Default run, no contextual/synthetic placeholders:

```bash
python scripts/train_models.py --cleaned-train data/processed/train_cleaned.csv --include-context-features false
```

Contextual simulation run:

```bash
python scripts/train_models.py --cleaned-train data/processed/train_cleaned.csv --include-context-features true
```

Local smoke test only:

```bash
python scripts/train_models.py --allow-raw-fallback --include-context-features false
```

The main target is `log1p(SalePrice)`. Metrics and residual plots are converted back to the original SalePrice scale.

## Outputs

Training writes:

- `outputs/metrics_summary.csv`
- `outputs/cv_results.csv`
- `outputs/ablation_results.csv`
- `outputs/feature_importance.csv`
- `outputs/residual_summary.csv`
- `outputs/error_by_price_bucket.csv`
- `outputs/error_by_neighborhood.csv`
- `outputs/error_by_overallqual.csv`
- `figures/actual_vs_predicted.png`
- `figures/residual_plot.png`
- `figures/residuals_vs_predicted.png`
- `figures/residual_distribution.png`
- `figures/error_by_price_bucket.png`
- `figures/top_feature_importance.png`
- `models/best_model.pkl`
- `models/final_pipeline.pkl`
- `submissions/kaggle_submission.csv`

`final_pipeline.pkl` includes feature engineering, preprocessing, and the trained model.

## Contextual Feature Placeholder

The pending synthetic/contextual feature block is in `src/house_price/config.py`:

```python
PENDING_CONTEXT_FEATURES = [
    "mortgage_rate",
    "days_on_market",
    "distance_to_center",
    # Add the 4th finalized contextual/synthetic column name here after EDA update.
]
```

These columns are normalized to snake case and included only with `--include-context-features true`. `days_on_market` should be used only if it is available at prediction time or clearly labeled as a contextual simulation.
