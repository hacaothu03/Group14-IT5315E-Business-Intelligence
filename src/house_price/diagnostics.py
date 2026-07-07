"""Diagnostic reports for feature engineering and model development."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vif_report(feature_frame: pd.DataFrame, max_features: int = 80) -> pd.DataFrame:
    """Compute VIF for numeric engineered features.

    VIF is computed on numeric, non-constant columns. When there are many numeric
    columns, the function prioritizes columns with highest absolute correlation
    to other numeric columns so the report stays focused on multicollinearity.
    """
    numeric = feature_frame.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if numeric.empty:
        return pd.DataFrame(columns=["feature", "vif", "r_squared", "n_features_used", "note"])

    numeric = numeric.fillna(numeric.median(numeric_only=True))
    nunique = numeric.nunique(dropna=False)
    numeric = numeric.loc[:, nunique > 1]
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["feature", "vif", "r_squared", "n_features_used", "note"])

    corr = numeric.corr().abs().fillna(0)
    priority = corr.where(~np.eye(len(corr), dtype=bool)).max().sort_values(ascending=False)
    selected = list(priority.head(max_features).index)
    numeric = numeric[selected]

    values = numeric.to_numpy(dtype=float)
    values = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    rows = []
    for idx, feature in enumerate(numeric.columns):
        y = values[:, idx]
        x = np.delete(values, idx, axis=1)
        x = np.column_stack([np.ones(x.shape[0]), x])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        y_hat = x @ coef
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        if r_squared >= 0.999999:
            vif = np.inf
        else:
            vif = 1.0 / max(1.0 - r_squared, 1e-12)
        rows.append(
            {
                "feature": feature,
                "vif": float(vif),
                "r_squared": float(r_squared),
                "n_features_used": int(numeric.shape[1]),
                "note": "high_multicollinearity" if vif >= 10 else ("moderate_multicollinearity" if vif >= 5 else "ok"),
            }
        )

    return pd.DataFrame(rows).sort_values(["vif", "feature"], ascending=[False, True])
