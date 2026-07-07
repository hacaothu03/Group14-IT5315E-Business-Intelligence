"""Evaluation metrics and residual summary tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from house_price import config


def to_price_scale(log_values: np.ndarray | pd.Series) -> np.ndarray:
    return np.expm1(np.asarray(log_values, dtype=float))


def regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    y_true = to_price_scale(y_true_log)
    y_pred = np.maximum(to_price_scale(y_pred_log), 0)
    ape = np.abs((y_true - y_pred) / np.maximum(y_true, 1))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(ape)),
        "r2": float(r2_score(y_true, y_pred)),
        "within_10_pct": float(np.mean(ape <= 0.10)),
        "within_20_pct": float(np.mean(ape <= 0.20)),
        "log_rmse": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
    }


def residual_frame(
    x: pd.DataFrame,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> pd.DataFrame:
    actual = to_price_scale(y_true_log)
    predicted = np.maximum(to_price_scale(y_pred_log), 0)
    residual = actual - predicted
    out = pd.DataFrame(
        {
            "actual": actual,
            "predicted": predicted,
            "residual": residual,
            "abs_error": np.abs(residual),
            "abs_pct_error": np.abs(residual) / np.maximum(actual, 1),
        },
        index=x.index,
    )
    normalized = x.copy()
    normalized.columns = [str(c).strip() for c in normalized.columns]
    for original, normalized_name in [("Neighborhood", "neighborhood"), ("OverallQual", "overall_qual")]:
        if original in normalized.columns:
            out[normalized_name] = normalized[original].values
        elif normalized_name in normalized.columns:
            out[normalized_name] = normalized[normalized_name].values
    out["price_bucket"] = pd.cut(
        out["actual"],
        bins=config.PRICE_BUCKETS,
        labels=config.PRICE_BUCKET_LABELS,
        include_lowest=True,
    )
    return out


def summarize_errors(residuals: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in residuals.columns:
        return pd.DataFrame()
    grouped = residuals.groupby(group_col, observed=False)
    return grouped.agg(
        n=("actual", "size"),
        actual_mean=("actual", "mean"),
        predicted_mean=("predicted", "mean"),
        residual_mean=("residual", "mean"),
        mae=("abs_error", "mean"),
        mape=("abs_pct_error", "mean"),
        within_10_pct=("abs_pct_error", lambda s: float((s <= 0.10).mean())),
        within_20_pct=("abs_pct_error", lambda s: float((s <= 0.20).mean())),
    ).reset_index()
