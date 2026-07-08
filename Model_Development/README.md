# Module 5 - Model Development

This module trains, evaluates, and packages the house price prediction models.

It uses cleaned data from Module 3 and shared feature engineering code in `src/house_price/`.

## Data Source

Primary inputs:

- train: `CleanData/data-v2/processed/train_cleaned.csv`
- test/submission: `CleanData/data-v2/processed/test_cleaned.csv`

The code does not modify `CleanData/`.

## Install

From the repository root:

```bash
pip install -r Model_Development/requirements.txt
```

From this folder:

```bash
pip install -r requirements.txt
```

`xgboost` is optional at runtime. If it is unavailable, the script trains the sklearn models and skips boosting gracefully.

## Train Models

From the repository root:

```bash
python Model_Development/scripts/train_models.py
```

From this folder:

```bash
python scripts/train_models.py
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

## Make Submission

```bash
python scripts/make_submission.py
```

## Outputs

Module 5 writes:

- `outputs/metrics_summary.csv`
- `outputs/cv_results.csv`
- `outputs/ablation_results.csv`
- `outputs/feature_importance.csv`
- `outputs/model_artifacts.csv`
- `outputs/tuning_results.csv`
- `outputs/residual_summary.csv`
- `outputs/error_by_price_bucket.csv`
- `outputs/error_by_neighborhood.csv`
- `outputs/error_by_overallqual.csv`
- `figures/actual_vs_predicted.png`
- `figures/residual_plot.png`
- `figures/residual_distribution.png`
- `figures/residuals_by_price_bucket.png`
- `figures/top_feature_importance.png`
- `models/best_model.pkl`
- `models/final_pipeline.pkl`
- `models/checkpoints/*_pipeline.pkl`
- `models/checkpoints/*_model.pkl`
- `submissions/kaggle_submission.csv`

`final_pipeline.pkl` is the selected best pipeline. Each file under `models/checkpoints/*_pipeline.pkl` is also a reusable full pipeline for that model family.

## Inference Smoke Test

From the repository root:

```bash
python Model_Development/src/smoke_test_inference.py
```

From this folder:

```bash
python src/smoke_test_inference.py
```

The inference path loads `models/final_pipeline.pkl`, applies feature engineering and preprocessing, predicts on the log scale, and converts to USD with `expm1`.

## For Module 6-7 Team

Read `reports/HANDOFF_TO_MODULE_6_7.md` first.

Use `src/predict.py` for inference and do not reimplement feature engineering in the API.
