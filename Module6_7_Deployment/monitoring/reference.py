"""Build the training-distribution reference dataset for drift monitoring.

The reference must be extracted the *same way* production predictions are logged
(``app/model_service.py``): raw monitored features come from the pre-pipeline
input, engineered features (``total_sf``, ``age_at_sale``, ...) come from the
fitted ``feature_engineering`` step, and ``predicted_sale_price`` is the model's
own prediction on the training data. ``actual_sale_price`` is the true label.

Run::

    python -m monitoring.reference      # from Module6_7_Deployment/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the deployment package importable when run as a script.
_DEPLOY_DIR = Path(__file__).resolve().parents[1]
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

from app.model_service import (  # noqa: E402
    ENGINEERED_MONITOR_FIELDS,
    RAW_MONITOR_FIELDS,
    get_service,
)
from monitoring.common import (  # noqa: E402
    ACTUAL_COL,
    CATEGORICAL_MONITOR_FEATURES,
    PREDICTION_COL,
    reference_path,
)


def build_reference(save: bool = True) -> pd.DataFrame:
    service = get_service()
    service.warm_up()
    artifacts = service.artifacts
    pipeline = artifacts["pipeline"]

    # Import normalize + config from the Module 4-5 package (already on sys.path
    # via model_service).
    from house_price import config
    from house_price.features import normalize_column_names

    train_path = config.CLEANDATA_PROCESSED_DIR / "train_cleaned.csv"
    train = normalize_column_names(pd.read_csv(train_path))

    if config.TARGET_COLUMN not in train.columns:
        raise ValueError(f"Training file missing target column '{config.TARGET_COLUMN}'.")
    actual = pd.to_numeric(train[config.TARGET_COLUMN], errors="coerce")

    # Model predictions on the training rows (log scale -> USD), same as serving.
    pred_log = pipeline.predict(train)
    predicted = np.maximum(np.expm1(pred_log), 0.0)

    # Engineered features from the fitted transformer (matches serving logs).
    fe = pipeline.named_steps.get("feature_engineering")
    engineered = fe.transform(train) if fe is not None else train

    out = pd.DataFrame(index=train.index)
    # Raw monitored features come straight from the (normalized) input.
    for col in RAW_MONITOR_FIELDS:
        if col in train.columns:
            out[col] = train[col].to_numpy()
    # Engineered monitored features come from the transformer output.
    for col in ENGINEERED_MONITOR_FIELDS:
        if col in engineered.columns:
            out[col] = engineered[col].to_numpy()
    for col in CATEGORICAL_MONITOR_FEATURES:
        if col in train.columns:
            out[col] = train[col].to_numpy()

    out[PREDICTION_COL] = predicted
    out[ACTUAL_COL] = actual.to_numpy()

    out = out.dropna(subset=[ACTUAL_COL]).reset_index(drop=True)

    if save:
        path = reference_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        print(f"Wrote reference dataset: {path}  ({len(out)} rows, {out.shape[1]} cols)")
    return out


def load_reference() -> pd.DataFrame:
    path = reference_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {path}. Run `python -m monitoring.reference` first."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    build_reference(save=True)
