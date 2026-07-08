# Group14 Business Intelligence - House Price Prediction

This repository is organized by project module.

- `Business_DataGen/`: Module 1 data generation/context work.
- `EDA/`: Module 2 exploratory data analysis.
- `CleanData/`: Module 3 data cleaning outputs.
- `Feature_Engineering/`: Module 4 feature engineering.
- `Model_Development/`: Module 5 model development.

Module 4 reads cleaned data from `CleanData/` and writes engineered feature outputs to `Feature_Engineering/outputs/`.

Module 5 reads the same cleaned data, trains/evaluates models, and writes model artifacts, figures, residual reports, and Kaggle submissions to `Model_Development/`.

Quick checks from the repo root:

```bash
python Feature_Engineering/scripts/run_feature_engineering.py
python Model_Development/src/smoke_test_inference.py
```
