"""Shared monitoring schema, baselines, and thresholds (Module 7).

The monitoring feature set matches exactly what ``app/model_service.py`` writes
to the prediction log, so production logs and the training reference are
directly comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Monitoring feature columns (must match model_service RAW/ENGINEERED fields).
NUMERIC_MONITOR_FEATURES = [
    "overall_qual",
    "gr_liv_area",
    "garage_cars",
    "total_bsmt_sf",
    "mortgage_rate",
    "days_on_market",
    "distance_to_center",
    "total_sf",
    "age_at_sale",
    "years_since_remodel",
    "total_bath",
]
CATEGORICAL_MONITOR_FEATURES = ["neighborhood"]

PREDICTION_COL = "predicted_sale_price"
ACTUAL_COL = "actual_sale_price"
TIMESTAMP_COL = "timestamp"

# PSI interpretation (industry-standard bands).
PSI_MODERATE = 0.10   # 0.10-0.25 => moderate shift (warning)
PSI_SIGNIFICANT = 0.25  # > 0.25   => significant drift

# Retraining-trigger thresholds (Module 5 handoff section 10).
MAPE_WARN = 0.10
MAPE_CRITICAL = 0.12
WITHIN10_WARN = 0.65        # validation baseline is 0.75
WITHIN10_CRITICAL = 0.55
N_DRIFTED_FEATURES_CRITICAL = 3


# Fallback validation baselines (lasso_v1) if metrics_summary.csv is unavailable.
_FALLBACK_BASELINE = {
    "valid_rmse": 18680.63,
    "valid_mae": 13195.73,
    "valid_mape": 0.07851366,
    "valid_within_10_pct": 0.75,
    "valid_within_20_pct": 0.93835616,
    "valid_log_rmse": 0.11031569,
}


def _modeling_src() -> Path:
    return Path(__file__).resolve().parents[2] / "Module4_5_Modeling" / "src"


def load_validation_baseline() -> dict[str, float]:
    """Read lasso_v1 validation metrics from Module 4-5 outputs (with fallback)."""
    try:
        src = str(_modeling_src())
        if src not in sys.path:
            sys.path.insert(0, src)
        import pandas as pd
        from house_price import config

        metrics = pd.read_csv(config.OUTPUT_DIR / "metrics_summary.csv")
        row = metrics.loc[metrics["model"].eq("lasso_regression")]
        if row.empty:
            row = metrics.head(1)
        r = row.iloc[0]
        return {
            "valid_rmse": float(r["valid_rmse"]),
            "valid_mae": float(r["valid_mae"]),
            "valid_mape": float(r["valid_mape"]),
            "valid_within_10_pct": float(r["valid_within_10_pct"]),
            "valid_within_20_pct": float(r["valid_within_20_pct"]),
            "valid_log_rmse": float(r["valid_log_rmse"]),
        }
    except Exception:
        return dict(_FALLBACK_BASELINE)


def default_log_path() -> Path:
    import os

    env = os.environ.get("PREDICTION_LOG_PATH")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "logs" / "predictions.jsonl"


def reference_path() -> Path:
    return Path(__file__).resolve().parent / "reference_data.csv"
