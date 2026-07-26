"""Inference utilities for Module 6 deployment.

The saved artifact design is Case A: ``models/final_pipeline.pkl`` is a full
sklearn Pipeline containing feature engineering, preprocessing, and the final
Lasso model. Do not refit encoders, scalers, or imputers at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from house_price import config
from house_price.features import normalize_column_names

MODEL_VERSION = "lasso_v1"
TARGET_TRANSFORM = "log1p"
INVERSE_TRANSFORM = "expm1"
DEFAULT_TRAIN_SCHEMA_PATH = config.CLEANDATA_PROCESSED_DIR / "train_cleaned.csv"
REQUIRED_CORE_FIELDS = [
    "neighborhood",
    "overall_qual",
    "gr_liv_area",
    "year_built",
    "year_remod_add",
]

# Columns the pipeline's FeatureEngineer derives from raw inputs *only when they
# are absent* (e.g. age_at_sale = yr_sold - year_built). They must NOT be
# pre-filled with training medians for partial inputs: doing so suppresses that
# per-record derivation, so a property with no explicit age would be scored at
# the median age (~35) regardless of its YearBuilt. Leaving them absent lets the
# pipeline compute them. Their source columns (year_built, year_remod_add,
# yr_sold) are guaranteed present via REQUIRED_CORE_FIELDS + date context. If a
# caller passes them explicitly, they are kept and respected.
PIPELINE_DERIVED_FIELDS = [
    "age_at_sale",
    "years_since_remodel",
]


def _load_metrics() -> dict[str, float | None]:
    path = config.OUTPUT_DIR / "metrics_summary.csv"
    if not path.exists():
        return {"approx_validation_mape": None}
    metrics = pd.read_csv(path)
    row = metrics.loc[metrics["model"].eq("lasso_regression")]
    if row.empty:
        row = metrics.head(1)
    return {"approx_validation_mape": float(row.iloc[0]["valid_mape"])}


def _load_training_defaults(schema_path: Path = DEFAULT_TRAIN_SCHEMA_PATH) -> tuple[list[str], dict[str, Any]]:
    if not schema_path.exists():
        raise FileNotFoundError(f"Training schema not found: {schema_path}")
    train = normalize_column_names(pd.read_csv(schema_path))
    train = train.drop(columns=["sale_price"], errors="ignore")
    defaults: dict[str, Any] = {}
    for col in train.columns:
        series = train[col]
        if pd.api.types.is_numeric_dtype(series):
            defaults[col] = float(series.median())
        else:
            mode = series.mode(dropna=True)
            defaults[col] = mode.iloc[0] if not mode.empty else "NoFeature"
    return list(train.columns), defaults


def load_artifacts(model_dir: str | Path | None = None) -> dict[str, Any]:
    model_dir_path = config.MODEL_DIR if model_dir is None or str(model_dir) == "models" else Path(model_dir)
    model_path = model_dir_path / "final_pipeline.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model pipeline not found: {model_path}")
    train_columns, defaults = _load_training_defaults()
    artifacts = {
        "pipeline": joblib.load(model_path),
        "model_version": MODEL_VERSION,
        "target_transform": TARGET_TRANSFORM,
        "inverse_transform": INVERSE_TRANSFORM,
        "train_columns": train_columns,
        "defaults": defaults,
        "artifact_case": "A_full_sklearn_pipeline",
    }
    artifacts.update(_load_metrics())
    return artifacts


def _as_dataframe(input_data: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()
    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])
    raise TypeError("input_data must be a dict or pandas.DataFrame")


def _apply_date_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has_valuation_date = "valuation_date" in df.columns
    has_search_date = {"search_year", "search_month"}.issubset(df.columns)
    has_training_date = {"yr_sold", "mo_sold"}.issubset(df.columns)

    if has_valuation_date:
        dates = pd.to_datetime(df["valuation_date"], errors="raise")
        df["yr_sold"] = dates.dt.year
        df["mo_sold"] = dates.dt.month
    elif has_search_date:
        df["yr_sold"] = pd.to_numeric(df["search_year"], errors="raise").astype(int)
        df["mo_sold"] = pd.to_numeric(df["search_month"], errors="raise").astype(int)
    elif not has_training_date:
        raise ValueError("Provide valuation_date or search_year/search_month for valuation context.")

    if ((df["mo_sold"] < 1) | (df["mo_sold"] > 12)).any():
        raise ValueError("search_month / MoSold must be between 1 and 12.")
    return df


def _prepare_input(input_data: dict[str, Any] | pd.DataFrame, artifacts: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    df = normalize_column_names(_as_dataframe(input_data))
    df = _apply_date_context(df)

    missing_required = [field for field in REQUIRED_CORE_FIELDS if field not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required raw field(s): {missing_required}")

    filled_fields: list[str] = []
    defaults = artifacts["defaults"]
    for col in artifacts["train_columns"]:
        if col in df.columns or col in PIPELINE_DERIVED_FIELDS:
            continue
        df[col] = defaults.get(col, np.nan)
        filled_fields.append(col)

    # Return columns in training order. Pipeline-derived fields that were left
    # unfilled above are recreated inside the pipeline's FeatureEngineer, so we
    # select only the columns actually present here.
    ordered = [col for col in artifacts["train_columns"] if col in df.columns]
    return df[ordered], filled_fields


def predict_price(
    input_data: dict[str, Any] | pd.DataFrame,
    artifacts: dict[str, Any] | None = None,
    return_details: bool = True,
) -> dict[str, Any] | pd.DataFrame:
    """Predict sale price in USD for one or many raw property records."""
    artifacts = artifacts or load_artifacts()
    prepared, filled_fields = _prepare_input(input_data, artifacts)
    pred_log = artifacts["pipeline"].predict(prepared)
    pred_price = np.maximum(np.expm1(pred_log), 0)

    if isinstance(input_data, dict):
        result = {
            "predicted_sale_price": float(pred_price[0]),
            "model_version": artifacts["model_version"],
            "target_transform": artifacts["target_transform"],
            "inverse_transform": artifacts["inverse_transform"],
            "approx_validation_mape": artifacts.get("approx_validation_mape"),
            "notes": "Prediction is for the provided valuation/search date context.",
        }
        if return_details:
            result["artifact_case"] = artifacts["artifact_case"]
            result["filled_default_fields"] = filled_fields
        return result

    output = pd.DataFrame(
        {
            "predicted_sale_price": pred_price.astype(float),
            "model_version": artifacts["model_version"],
        },
        index=prepared.index,
    )
    if return_details:
        output["target_transform"] = artifacts["target_transform"]
        output["inverse_transform"] = artifacts["inverse_transform"]
    return output
