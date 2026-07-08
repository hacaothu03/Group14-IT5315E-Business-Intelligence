# Module 4 - Feature Engineering

This module creates engineered features for the house price prediction pipeline.

It uses cleaned data from Module 3:

- `CleanData/data-v2/processed/train_cleaned.csv`
- `CleanData/data-v2/processed/test_cleaned.csv` when needed downstream

The code does not modify `CleanData/`.

## Run

From the repository root:

```bash
python Feature_Engineering/scripts/run_feature_engineering.py
```

From this folder:

```bash
python scripts/run_feature_engineering.py
```

## Outputs

Module 4 writes:

- `outputs/engineered_train_full.csv`
- `outputs/feature_list_full.csv`
- `outputs/feature_list_reduced_linear.csv`
- `outputs/vif_report.csv`

## Notes

The reusable feature engineering code is shared with Module 5 under:

```text
Model_Development/src/house_price/
```

`run_feature_engineering.py` sets the module output root to `Feature_Engineering/`, so feature outputs stay separated from model development outputs.
